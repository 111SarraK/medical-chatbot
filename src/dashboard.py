import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import json
import time
from .database import DatabaseManager
from dotenv import load_dotenv


# Charger les variables d'environnement
load_dotenv()

def format_feedback_score(score):
    """Convertit le score numérique en texte."""
    if score >= 0.8:
        return "Très positif"
    elif score >= 0.6:
        return "Positif"
    elif score >= 0.4:
        return "Neutre"
    elif score >= 0.2:
        return "Négatif"
    else:
        return "Très négatif"

def load_data():
    """Charge les données de feedback depuis la base de données."""
    db_manager = DatabaseManager(
        instance_connection_name=os.getenv("INSTANCE_CONNECTION_NAME"),
        db_user=os.getenv("DB_USER"),
        db_pass=os.getenv("DB_PASS"),
        db_name=os.getenv("DB_NAME")
    )
    db_manager.initialize_connection()
    
    # Récupérer les feedbacks
    feedbacks = db_manager.get_all_feedbacks()
    
    # Convertir en DataFrame pandas
    if feedbacks:
        df = pd.DataFrame(feedbacks)
        # Convertir les timestamps en datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Ajouter une colonne pour les catégories de feedback
        df['feedback_category'] = df['feedback_score'].apply(format_feedback_score)
        return df
    else:
        # Retourner un DataFrame vide avec les colonnes attendues
        return pd.DataFrame(columns=['feedback_id', 'question', 'feedback_score', 'timestamp', 'feedback_category'])

def main():
    """Application principale du dashboard."""
    st.set_page_config(
        page_title="Dashboard - Feedback du Chatbot Médical",
        page_icon="📊",
        layout="wide"
    )
    
    # Titre et description
    st.title("📊 Dashboard - Feedback du Chatbot Médical")
    st.write("Visualisation et analyse des feedbacks utilisateurs pour le chatbot médical.")
    
    # Chargement des données avec mise en cache
    @st.cache_data(ttl=300)  # Cache de 5 minutes
    def get_cached_data():
        return load_data()
    
    # Bouton de rafraîchissement
    if st.button("🔄 Rafraîchir les données"):
        st.cache_data.clear()
    
    df = get_cached_data()
    
    if df.empty:
        st.warning("Aucun feedback disponible pour le moment.")
        return
    
    # Section des KPIs en haut
    st.header("Indicateurs clés de performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_score = df['feedback_score'].mean()
        st.metric(
            label="Score moyen",
            value=f"{avg_score:.2f}/1.0",
            delta=f"{(avg_score - 0.5) * 2:.1f}%",
            delta_color="normal"
        )
    
    with col2:
        total_feedbacks = len(df)
        st.metric(
            label="Total des feedbacks",
            value=total_feedbacks
        )
    
    with col3:
        recent_feedbacks = df[df['timestamp'] > (datetime.now() - timedelta(days=7))].shape[0]
        st.metric(
            label="Feedbacks cette semaine",
            value=recent_feedbacks
        )
    
    with col4:
        positive_pct = df[df['feedback_score'] >= 0.6].shape[0] / max(total_feedbacks, 1) * 100
        st.metric(
            label="% de feedbacks positifs",
            value=f"{positive_pct:.1f}%",
            delta=f"{positive_pct - 50:.1f}%" if positive_pct != 50 else None
        )
    
    # Graphiques
    st.header("Analyse des feedbacks")
    
    tab1, tab2, tab3 = st.tabs(["Tendances temporelles", "Distribution des scores", "Questions fréquentes"])
    
    with tab1:
        # Tendance temporelle des feedbacks
        df_daily = df.resample('D', on='timestamp')['feedback_score'].mean().reset_index()
        df_daily = df_daily.rename(columns={'feedback_score': 'score_moyen'})
        
        fig = px.line(
            df_daily, 
            x='timestamp', 
            y='score_moyen',
            title="Évolution du score moyen des feedbacks dans le temps",
            labels={"timestamp": "Date", "score_moyen": "Score moyen"},
            markers=True
        )
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume de feedbacks par jour
        df_count = df.resample('D', on='timestamp').size().reset_index(name='nombre_feedbacks')
        
        fig2 = px.bar(
            df_count,
            x='timestamp',
            y='nombre_feedbacks',
            title="Volume quotidien de feedbacks",
            labels={"timestamp": "Date", "nombre_feedbacks": "Nombre de feedbacks"}
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        # Distribution des scores de feedback
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogramme des scores
            fig3 = px.histogram(
                df,
                x='feedback_score',
                nbins=10,
                title="Distribution des scores de feedback",
                labels={"feedback_score": "Score de feedback", "count": "Nombre"}
            )
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            # Camembert des catégories de feedback
            category_counts = df['feedback_category'].value_counts().reset_index()
            category_counts.columns = ['Catégorie', 'Nombre']
            
            fig4 = px.pie(
                category_counts,
                values='Nombre',
                names='Catégorie',
                title="Répartition des catégories de feedback",
                hole=0.4
            )
            st.plotly_chart(fig4, use_container_width=True)
    
    with tab3:
        # Analyse des questions fréquentes
        st.subheader("Questions ayant reçu le plus de feedbacks")
        
        # Regrouper par question et calculer le nombre et le score moyen
        question_stats = df.groupby('question').agg({
            'feedback_score': ['mean', 'count']
        }).reset_index()
        question_stats.columns = ['Question', 'Score moyen', 'Nombre de feedbacks']
        question_stats = question_stats.sort_values('Nombre de feedbacks', ascending=False)
        
        # Filtrer pour les questions avec au moins 3 caractères
        question_stats = question_stats[question_stats['Question'].str.len() > 3]
        
        # Afficher les 10 premières questions
        st.dataframe(
            question_stats.head(10),
            column_config={
                "Score moyen": st.column_config.ProgressColumn(
                    "Score moyen",
                    format="%.2f",
                    min_value=0,
                    max_value=1
                )
            },
            use_container_width=True
        )
        
        # Visualisation de la relation entre le nombre de feedbacks et le score moyen
        fig5 = px.scatter(
            question_stats.head(20),
            x='Nombre de feedbacks',
            y='Score moyen',
            text='Question',
            title="Relation entre popularité et satisfaction des questions",
            size='Nombre de feedbacks',
            color='Score moyen',
            color_continuous_scale='RdYlGn'
        )
        fig5.update_traces(
            textposition='top center',
            marker=dict(sizemode='area', sizeref=2.*max(question_stats['Nombre de feedbacks'].head(20))/(40.**2))
        )
        fig5.update_layout(height=600)
        st.plotly_chart(fig5, use_container_width=True)
    
    # Filtres et exploration détaillée
    st.header("Exploration détaillée des feedbacks")
    
    # Filtres
    col1, col2, col3 = st.columns(3)
    with col1:
        # Filtre par période
        period_options = ["Tous", "Aujourd'hui", "Cette semaine", "Ce mois", "Les 3 derniers mois"]
        selected_period = st.selectbox("Période", period_options)
    
    with col2:
        # Filtre par catégorie de feedback
        categories = ["Tous"] + sorted(df['feedback_category'].unique().tolist())
        selected_category = st.selectbox("Catégorie de feedback", categories)
    
    with col3:
        # Filtre par texte de recherche
        search_text = st.text_input("Rechercher dans les questions", "")
    
    # Appliquer les filtres
    filtered_df = df.copy()
    
    if selected_period != "Tous":
        if selected_period == "Aujourd'hui":
            filtered_df = filtered_df[filtered_df['timestamp'] >= datetime.now().replace(hour=0, minute=0, second=0)]
        elif selected_period == "Cette semaine":
            filtered_df = filtered_df[filtered_df['timestamp'] >= (datetime.now() - timedelta(days=7))]
        elif selected_period == "Ce mois":
            filtered_df = filtered_df[filtered_df['timestamp'] >= (datetime.now() - timedelta(days=30))]
        elif selected_period == "Les 3 derniers mois":
            filtered_df = filtered_df[filtered_df['timestamp'] >= (datetime.now() - timedelta(days=90))]
    
    if selected_category != "Tous":
        filtered_df = filtered_df[filtered_df['feedback_category'] == selected_category]
    
    if search_text:
        filtered_df = filtered_df[filtered_df['question'].str.contains(search_text, case=False)]
    
    # Affichage des feedbacks filtrés
    st.subheader(f"Feedbacks ({len(filtered_df)} résultats)")
    
    if not filtered_df.empty:
        st.dataframe(
            filtered_df[['timestamp', 'question', 'feedback_score', 'feedback_category']].sort_values('timestamp', ascending=False),
            column_config={
                "timestamp": st.column_config.DatetimeColumn("Date et heure", format="DD/MM/YYYY HH:mm"),
                "question": "Question",
                "feedback_score": st.column_config.ProgressColumn("Score", min_value=0, max_value=1, format="%.2f"),
                "feedback_category": "Catégorie"
            },
            use_container_width=True
        )
        
        # Export des données
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Télécharger les données filtrées (CSV)",
            data=csv,
            file_name=f"feedback_chatbot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("Aucun résultat ne correspond aux filtres sélectionnés.")

if __name__ == "__main__":
    main()