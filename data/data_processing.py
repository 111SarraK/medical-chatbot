import os
import json
import pandas as pd
from typing import List, Dict
import re
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import cv2
import numpy as np
from PIL import Image

def process_layoutlm_dataset(dataset_path: str) -> List[Dict]:

    processed_documents = []
    
    # Process PDF documents if present
    pdf_dir = os.path.join(dataset_path, "pdfs")
    if os.path.exists(pdf_dir):
        pdf_loader = DirectoryLoader(
            pdf_dir,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        
        try:
            pdf_docs = pdf_loader.load()
            # Extract relevant information
            for doc in pdf_docs:
                source = os.path.basename(doc.metadata["source"])
                focus_area = determine_focus_area(doc.page_content)
                
                # Split long documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = text_splitter.split_text(doc.page_content)
                
                for i, chunk in enumerate(chunks):
                    processed_documents.append({
                        "text": chunk,
                        "source": f"{source}_chunk_{i}",
                        "focus_area": focus_area,
                        "has_layout": False
                    })
        except Exception as e:
            print(f"Error processing PDF documents: {e}")
    
    # Process JSON files with annotations (LayoutLM specific)
    annotations_dir = os.path.join(dataset_path, "annotations")
    if os.path.exists(annotations_dir):
        for root, _, files in os.walk(annotations_dir):
            for file in files:
                if file.endswith(".json"):
                    try:
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            annotation_data = json.load(f)
                        
                        # Extract information from annotation data
                        source = os.path.basename(file).replace('.json', '')
                        
                        # Process the annotation based on LayoutLM format
                        processed_doc = process_layoutlm_annotation(annotation_data, source, dataset_path)
                        if processed_doc:
                            processed_documents.append(processed_doc)
                            
                    except Exception as e:
                        print(f"Error processing annotation file {file}: {e}")
    
    # Process OCR results if present
    ocr_dir = os.path.join(dataset_path, "ocr_results")
    if os.path.exists(ocr_dir):
        for root, _, files in os.walk(ocr_dir):
            for file in files:
                if file.endswith(".json"):
                    try:
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r') as f:
                            ocr_data = json.load(f)
                        
                        source = os.path.basename(file).replace('.json', '')
                        text = extract_text_from_ocr(ocr_data)
                        focus_area = determine_focus_area(text)
                        
                        # Find corresponding image if exists
                        image_extensions = ['.jpg', '.jpeg', '.png']
                        image_path = None
                        for ext in image_extensions:
                            img_path = os.path.join(dataset_path, "images", f"{source}{ext}")
                            if os.path.exists(img_path):
                                image_path = img_path
                                break
                        
                        has_layout = image_path is not None and 'bbox' in ocr_data
                        
                        processed_documents.append({
                            "text": text,
                            "source": source,
                            "focus_area": focus_area,
                            "has_layout": has_layout,
                            "image_path": image_path,
                            "layout_info": extract_layout_info(ocr_data) if has_layout else None
                        })
                    except Exception as e:
                        print(f"Error processing OCR file {file}: {e}")
    
    # Process images with form understanding results (specific to LayoutLM)
    forms_dir = os.path.join(dataset_path, "forms")
    if os.path.exists(forms_dir):
        for root, _, files in os.walk(forms_dir):
            for file in files:
                if file.endswith(".json"):
                    try:
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r') as f:
                            form_data = json.load(f)
                        
                        source = os.path.basename(file).replace('.json', '')
                        
                        # Find corresponding image
                        image_extensions = ['.jpg', '.jpeg', '.png']
                        image_path = None
                        for ext in image_extensions:
                            img_path = os.path.join(dataset_path, "images", f"{source}{ext}")
                            if os.path.exists(img_path):
                                image_path = img_path
                                break
                        
                        if image_path:
                            # Extract text and layout information from form data
                            text = extract_text_from_form(form_data)
                            focus_area = determine_focus_area(text)
                            layout_info = extract_layout_info_from_form(form_data)
                            
                            processed_documents.append({
                                "text": text,
                                "source": source,
                                "focus_area": focus_area,
                                "has_layout": True,
                                "image_path": image_path,
                                "layout_info": layout_info
                            })
                    except Exception as e:
                        print(f"Error processing form file {file}: {e}")
    
    # Process Q&A pairs if available
    qa_path = os.path.join(dataset_path, "qa_pairs.csv")
    if os.path.exists(qa_path):
        try:
            qa_df = pd.read_csv(qa_path)
            
            # Create test data for evaluation
            test_data = []
            for _, row in qa_df.iterrows():
                test_data.append({
                    "question": row["question"],
                    "answer": row["answer"]
                })
            
            # Save test data for evaluation
            os.makedirs("data/processed", exist_ok=True)
            with open("data/processed/test_data.json", "w") as f:
                json.dump(test_data, f, indent=2)
                
        except Exception as e:
            print(f"Error processing Q&A pairs: {e}")
    
    return processed_documents

def process_layoutlm_annotation(annotation_data: Dict, source: str, dataset_path: str) -> Dict:
    """
    Process LayoutLM annotation data.
    
    Args:
        annotation_data: The annotation data from JSON file
        source: Source identifier
        dataset_path: Base path to the dataset
        
    Returns:
        Processed document with layout information
    """
    # Extract text from annotations
    text = ""
    words = []
    boxes = []
    
    if 'form' in annotation_data:
        # FUNSD-like format
        for item in annotation_data['form']:
            if 'text' in item:
                text += item['text'] + " "
            if 'words' in item:
                for word_item in item['words']:
                    if 'text' in word_item and 'box' in word_item:
                        words.append(word_item['text'])
                        boxes.append(word_item['box'])
    elif 'words' in annotation_data and 'bbox' in annotation_data:
        # Direct words and bounding boxes
        text = " ".join(annotation_data['words'])
        words = annotation_data['words']
        boxes = annotation_data['bbox']
    
    # Find corresponding image
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_path = None
    for ext in image_extensions:
        img_path = os.path.join(dataset_path, "images", f"{source}{ext}")
        if os.path.exists(img_path):
            image_path = img_path
            break
    
    focus_area = determine_focus_area(text)
    
    return {
        "text": text,
        "source": source,
        "focus_area": focus_area,
        "has_layout": True,
        "image_path": image_path,
        "layout_info": {
            "words": words,
            "boxes": boxes
        }
    }

def determine_focus_area(text: str) -> str:
    """
    Determine the medical focus area based on text content.
    
    This is a simplified approach. A more sophisticated method would
    use medical ontologies or classification models.
    """
    # Simple keyword-based categorization
    focus_areas = {
        "cardiology": ["heart", "cardiac", "cardiovascular", "blood pressure", "hypertension"],
        "neurology": ["brain", "neural", "neuron", "cognitive", "nerve"],
        "oncology": ["cancer", "tumor", "oncology", "malignant", "chemotherapy"],
        "pediatrics": ["child", "infant", "pediatric", "adolescent"],
        "dermatology": ["skin", "dermatology", "rash", "acne"],
        "general": ["health", "wellness", "medicine", "treatment"],
        "laboratory": ["lab", "test", "blood", "sample", "specimen"],
        "pharmacy": ["drug", "medication", "dose", "prescription"],
        "radiology": ["x-ray", "scan", "mri", "ct", "imaging"],
        "surgery": ["surgery", "operation", "surgical", "procedure"]
    }
    
    text_lower = text.lower()
    
    for area, keywords in focus_areas.items():
        for keyword in keywords:
            if keyword in text_lower:
                return area
    
    return "uncategorized"

def extract_text_from_ocr(ocr_data: Dict) -> str:
    """
    Extract text from OCR data structure based on LayoutLM format.
    """
    if isinstance(ocr_data, str):
        return ocr_data
    
    if "text" in ocr_data:
        return ocr_data["text"]
    
    if "words" in ocr_data:
        if isinstance(ocr_data["words"], list):
            return " ".join(ocr_data["words"])
    
    if "pages" in ocr_data:
        texts = []
        for page in ocr_data["pages"]:
            if "text" in page:
                texts.append(page["text"])
            elif "lines" in page:
                for line in page["lines"]:
                    if "text" in line:
                        texts.append(line["text"])
        return " ".join(texts)
    
    # If structure is unknown, try to extract any text content
    ocr_str = json.dumps(ocr_data)
    # Find anything that looks like text content
    text_elements = re.findall(r'"text"\s*:\s*"([^"]+)"', ocr_str)
    
    return " ".join(text_elements)

def extract_text_from_form(form_data: Dict) -> str:
    """
    Extract text from form data.
    """
    if "text" in form_data:
        return form_data["text"]
    
    texts = []
    
    # FUNSD format
    if "form" in form_data:
        for item in form_data["form"]:
            if "text" in item:
                texts.append(item["text"])
            elif "words" in item:
                words = [word["text"] for word in item["words"] if "text" in word]
                texts.append(" ".join(words))
    
    # Handle other possible structures
    if "entities" in form_data:
        for entity in form_data["entities"]:
            if "text" in entity:
                texts.append(entity["text"])
    
    return " ".join(texts)

def extract_layout_info(ocr_data: Dict) -> Dict:
    """
    Extract layout information from OCR data.
    """
    layout_info = {
        "words": [],
        "boxes": []
    }
    
    if "words" in ocr_data and "bbox" in ocr_data:
        layout_info["words"] = ocr_data["words"]
        layout_info["boxes"] = ocr_data["bbox"]
    elif "pages" in ocr_data:
        for page in ocr_data["pages"]:
            if "words" in page and "bbox" in page:
                layout_info["words"].extend(page["words"])
                layout_info["boxes"].extend(page["bbox"])
    
    return layout_info

def extract_layout_info_from_form(form_data: Dict) -> Dict:
    """
    Extract layout information from form data.
    """
    words = []
    boxes = []
    
    # FUNSD format
    if "form" in form_data:
        for item in form_data["form"]:
            if "words" in item:
                for word in item["words"]:
                    if "text" in word and "box" in word:
                        words.append(word["text"])
                        boxes.append(word["box"])
    
    # Direct format
    if "words" in form_data and "bbox" in form_data:
        words = form_data["words"]
        boxes = form_data["bbox"]
    
    return {
        "words": words,
        "boxes": boxes
    }

def preprocess_image_for_layoutlm(image_path: str) -> Dict:
    """
    Preprocess image for LayoutLM.
    
    Args:
        image_path: Path to the image
        
    Returns:
        Dict with preprocessed image data
    """
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        # Convert to RGB format (LayoutLM expects RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image if too large (LayoutLM has input size limits)
        max_dim = 1000
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)))
        
        # Convert to PIL image for compatibility with transformers
        pil_image = Image.fromarray(image)
        
        return {
            "image": pil_image,
            "width": pil_image.width,
            "height": pil_image.height
        }
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None