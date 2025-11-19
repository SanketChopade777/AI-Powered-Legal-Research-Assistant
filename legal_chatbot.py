import streamlit as st
from utils.legal_chatbot.rag_pipeline import (
    retrieve_docs,
    answer_query_with_fallback
)
from utils.legal_chatbot.memory_manager import get_memory_manager
from vector_database import train_on_articles, load_vector_store
from config import PRETRAINED_DB_PATH
import os
import json
from datetime import datetime
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu

# Custom CSS for beautiful dark theme styling
def inject_custom_css():
    st.markdown("""
    <style>
    /* Main background - Dark theme */
    .stApp {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #ffffff;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
        color: white !important;
        border-right: 2px solid #4cc9f0;
    }

    /* Sidebar text color */
    .css-1d391kg * {
        color: white !important;
    }

    /* Main content area */
    .main .block-container {
        background: rgba(26, 26, 46, 0.9);
        border-radius: 20px;
        padding: 30px;
        margin: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
        min-height: 80vh;
        border: 1px solid #4cc9f0;
    }

    /* Chat containers */
    .stChatMessage {
        border-radius: 20px !important;
        margin: 15px 0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        border: none !important;
    }

    /* User message (Odd) - Darker, contrasting color */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background: linear-gradient(135deg, #1f286b 0%, #151e3f 100%) !important; /* Deep Indigo to Dark Navy */
        color: white !important;
        border: none;
    }
    
    /* Assistant message (Even) - Lighter, professional color */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background: linear-gradient(135deg, #4f7cac 0%, #3e6080 100%) !important; /* Slate Blue to Gray-Blue */
        color: white !important;
        border: none;
    }
    
    /* Chat message content */
    .stChatMessageContent {
        padding: 20px !important;
    }

    /* Buttons styling */
    .stButton>button {
        border-radius: 20px;
        border: none;
        background: linear-gradient(135deg, #f72585 0%, #b5179e 100%);
        color: white;
        font-weight: bold;
        padding: 12px 24px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }

    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(247, 37, 133, 0.4);
        background: linear-gradient(135deg, #b5179e 0%, #7209b7 100%);
    }

    /* Feedback buttons */
    .feedback-btn {
        background: linear-gradient(135deg, #4cc9f0 0%, #4895ef 100%) !important;
        margin: 8px;
        padding: 8px 16px !important;
    }

    /* Input field */
    .stTextInput>div>div>input {
        border-radius: 25px;
        border: 2px solid #4cc9f0;
        padding: 15px 20px;
        font-size: 18px;
        background: rgba(26, 26, 46, 0.8);
        color: white;
    }

    /* Navigation button */
    .nav-btn {
        background: linear-gradient(135deg, #4cc9f0 0%, #4361ee 100%) !important;
        margin: 10px 0;
        width: 100%;
    }

    /* Remove file uploader related styles since we're removing that functionality */
    </style>
    """, unsafe_allow_html=True)

def render_back_button():
    """Render the back button to return to home"""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("← Back to Home", key="doc_back_btn", use_container_width=True):
            if 'current_page' in st.session_state:
                st.session_state.current_page = 'home'
                st.rerun()

def initialize_pretrained_db():
    if not os.path.exists(PRETRAINED_DB_PATH):
        st.info("Initializing legal knowledge base...")
        train_on_articles()
    return load_vector_store()

def load_chat_history():
    """Load chat history from session state"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    return st.session_state.chat_history

def save_chat_message(role, message, feedback=None):
    """Save chat message to history"""
    chat_entry = {
        'timestamp': datetime.now().isoformat(),
        'role': role,
        'message': message,
        'feedback': feedback
    }
    st.session_state.chat_history.append(chat_entry)

def save_feedback(feedback, message_index):
    """Save user feedback for a specific message"""
    if 0 <= message_index < len(st.session_state.chat_history):
        st.session_state.chat_history[message_index]['feedback'] = feedback
        st.success("✅ Feedback saved!")

def export_chat_history():
    """Export chat history to JSON"""
    if st.session_state.chat_history:
        return json.dumps(st.session_state.chat_history, indent=2)
    return None

def analyze_feedback():
    """Analyze feedback statistics with detailed metrics"""
    if not st.session_state.chat_history:
        return None

    feedbacks = [msg.get('feedback') for msg in st.session_state.chat_history
                 if msg.get('feedback') and msg['role'] == 'assistant']

    if not feedbacks:
        return None

    # Count ratings and collect data
    rating_counts = {'👍 Good': 0, '👎 Needs Improvement': 0, '📝 With Notes': 0}
    comments = []

    for feedback in feedbacks:
        if isinstance(feedback, dict):
            rating = feedback.get('rating', 'neutral')
            if rating == 'good':
                rating_counts['👍 Good'] += 1
            elif rating == 'bad':
                rating_counts['👎 Needs Improvement'] += 1
            elif rating == 'neutral':
                rating_counts['📝 With Notes'] += 1

            comment = feedback.get('comment', '')
            if comment:
                comments.append(comment)

    return {
        'total_feedback': len(feedbacks),
        'rating_distribution': rating_counts,
        'total_messages': len(st.session_state.chat_history),
        'user_messages': len([m for m in st.session_state.chat_history if m['role'] == 'user']),
        'ai_messages': len([m for m in st.session_state.chat_history if m['role'] == 'assistant']),
        'recent_comments': comments[-3:] if comments else [],
        'feedback_ratio': len(feedbacks) / len([m for m in st.session_state.chat_history if
                                                m['role'] == 'assistant']) * 100 if st.session_state.chat_history else 0
    }

def create_feedback_chart(feedback_data):
    """Create beautiful charts for feedback analysis"""
    if not feedback_data:
        return None

    # Create pie chart for ratings
    ratings_df = pd.DataFrame({
        'Rating': list(feedback_data['rating_distribution'].keys()),
        'Count': list(feedback_data['rating_distribution'].values())
    })

    fig_pie = px.pie(ratings_df, values='Count', names='Rating',
                     title='📊 Feedback Distribution',
                     color_discrete_sequence=['#4cc9f0', '#4361ee', '#7209b7'])
    fig_pie.update_traces(textposition='inside', textinfo='percent+label',
                          marker=dict(line=dict(color='#1a1a2e', width=2)))
    fig_pie.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )

    return fig_pie

def show_analytics():
    """Show analytics content"""
    st.markdown("""
    <div class='beautiful-card'>
        <h2>📈 Performance Analytics</h2>
        <p>Track your interaction metrics and feedback patterns</p>
    </div>
    """, unsafe_allow_html=True)

    feedback_data = analyze_feedback()

    if feedback_data:
        # Metrics in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("💬 Total Messages", feedback_data['total_messages'])
        with col2:
            st.metric("⭐ Feedback Received", feedback_data['total_feedback'])
        with col3:
            st.metric("📊 Feedback Ratio", f"{feedback_data['feedback_ratio']:.1f}%")

        # Chart
        fig = create_feedback_chart(feedback_data)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # Recent comments
        if feedback_data['recent_comments']:
            st.markdown("### 💬 Recent Feedback Comments")
            for i, comment in enumerate(feedback_data['recent_comments']):
                st.markdown(f"""
                <div class='beautiful-card'>
                    <strong>Comment #{i + 1}:</strong> "{comment}"
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("🌟 No feedback data yet. Start chatting and provide feedback to see analytics!")

def show_settings():
    """Show settings content"""
    st.markdown("""
    <div class='beautiful-card'>
        <h2>⚙️ System Settings</h2>
        <p>Configure your Legal Assistant preferences</p>
    </div>
    """, unsafe_allow_html=True)

    # Model management
    st.markdown("### 🧠 Knowledge Base Management")
    if st.button("🔄 Update Knowledge Base", help="Refresh the AI's legal knowledge with latest articles"):
        with st.spinner("🔄 Updating knowledge base..."):
            try:
                train_on_articles()
                st.session_state.pretrained_db = load_vector_store()
                st.success("✅ Knowledge base updated successfully!")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    # Chat history management
    st.markdown("### 💾 Chat History")
    if st.session_state.chat_history:
        st.info(f"📊 You have {len(st.session_state.chat_history)} messages in your chat history")

        # Export button
        chat_json = export_chat_history()
        if chat_json:
            st.download_button(
                label="📥 Export Chat History",
                data=chat_json,
                file_name=f"legal_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                help="Download your complete conversation history"
            )

        # Clear history button
        if st.button("🗑️ Clear All Chat History", help="Start a fresh conversation"):
            st.session_state.chat_history = []
            st.success("✅ Chat history cleared!")
            st.rerun()
    else:
        st.info("💭 No chat history yet. Start a conversation to see options here.")

def show_chat():
    """Show chat content"""
    # Display chat messages with beautiful styling
    for i, chat in enumerate(st.session_state.chat_history):
        if chat['role'] == 'user':
            with st.chat_message("user"):
                st.markdown(f"**You:**\n{chat['message']}")
        else:
            with st.chat_message("assistant", avatar="⚖️"):
                st.markdown(f"**Assistant:**\n{chat['message']}")

                # Feedback buttons
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    if st.button("👍", key=f"good_{i}", help="Good response", use_container_width=True):
                        save_feedback({"rating": "good", "comment": "User liked the response"}, i)
                with col2:
                    if st.button("👎", key=f"bad_{i}", help="Needs improvement", use_container_width=True):
                        save_feedback({"rating": "bad", "comment": "User disliked the response"}, i)
                with col3:
                    if st.button("💬 Add Note", key=f"note_{i}", help="Add specific feedback", use_container_width=True):
                        note = st.text_input("Your note:", key=f"note_input_{i}", label_visibility="collapsed")
                        if note:
                            save_feedback({"rating": "neutral", "comment": note}, i)

def show_right_panel_content(selected_tab):
    """Show content in the right panel based on selected tab"""
    if selected_tab == "📈 Analytics":
        pass
    elif selected_tab == "⚙️ Settings":
        pass
    else:
        # Quick question suggestions
        st.markdown("""
        <div class='beautiful-card'>
            <h4>💡 Quick Questions</h4>
            <p>Try asking:</p>
            <ul>
            <li>What are my rights in a rental dispute?</li>
            <li>How to file a small claims case?</li>
            <li>Explain contract termination clauses</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # Quick actions for chat tab
        st.markdown("""
        <div class='beautiful-card'>
            <h3>🚀 Quick Actions</h3>
            <p>Manage your chat session</p>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.chat_history:
            st.info(f"💬 {len([m for m in st.session_state.chat_history if m['role'] == 'user'])} questions asked")
            st.info(f"⚖️ {len([m for m in st.session_state.chat_history if m['role'] == 'assistant'])} responses given")

            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        else:
            st.info("💭 Start chatting to see quick actions here")

def show_left_panel_content(selected_tab):
    """Show content in the left panel based on selected tab"""
    if selected_tab == "💬 Chat":
        show_chat()

        # Chat input at bottom
        user_query = st.chat_input("💭 Ask your legal question here...")

        if user_query:
            save_chat_message('user', user_query)
            with st.spinner("🔍 Analyzing your question..."):
                try:
                    # Removed document upload functionality - only use pretrained DB
                    retrieved_docs = retrieve_docs(
                        user_query,
                        st.session_state.pretrained_db
                    )
                    response = answer_query_with_fallback(
                        retrieved_docs,
                        user_query,
                        st.session_state.memory_manager
                    )

                    save_chat_message('assistant', response)
                    st.rerun()

                except Exception as e:
                    error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
                    save_chat_message('assistant', error_msg)
                    st.rerun()

    elif selected_tab == "📈 Analytics":
        show_analytics()

    elif selected_tab == "⚙️ Settings":
        show_settings()

def main():
    # st.set_page_config(
    #     page_title="AI Legal Assistant - Chat",
    #     layout="wide",
    #     page_icon="⚖️",
    #     initial_sidebar_state="expanded"
    # )

    # Inject custom CSS
    inject_custom_css()

    # Add back button at the top
    render_back_button()


    # Initialize session state
    if 'pretrained_db' not in st.session_state:
        st.session_state.pretrained_db = initialize_pretrained_db()
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "Chat"
    if 'memory_manager' not in st.session_state:
        st.session_state.memory_manager = get_memory_manager(session_id="user_session")

    # Initialize chat history
    load_chat_history()

    # Sidebar navigation
    with st.sidebar:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #4361ee 0%, #3a0ca3 100%); 
                   padding: 25px; border-radius: 20px; color: white; text-align: center; margin-bottom: 20px;'>
            <h1>⚖️ LegalMind AI</h1>
            <p style='margin: 0;'>Your Intelligent Legal Assistant</p>
        </div>
        """, unsafe_allow_html=True)

        # # Navigation back to main page
        # if st.button("🏠 Back to Main Menu", use_container_width=True, key="nav_home"):
        #     webbrowser.open_new_tab("http://localhost:8501")

        # Navigation menu
        selected_tab = option_menu(
            menu_title=None,
            options=["💬 Chat", "📈 Analytics", "⚙️ Settings"],
            icons=["chat", "bar-chart", "gear"],
            default_index=0,
            styles={
                "container": {"padding": "0", "background-color": "transparent"},
                "icon": {"color": "white", "font-size": "20px"},
                "nav-link": {
                    "font-size": "18px",
                    "text-align": "left",
                    "margin": "5px 0",
                    "border-radius": "15px",
                    "padding": "15px",
                    "background": "rgba(255,255,255,0.1)"
                },
                "nav-link-selected": {
                    "background": "linear-gradient(135deg, #4cc9f0 0%, #4361ee 100%)",
                    "color": "white"
                },
            }
        )

        st.markdown("---")

        # Removed file upload section completely
        st.markdown("### 🎯 Chat Mode")
        st.info("💬 You're in **Legal Expert** mode. Ask any legal questions!")

        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: rgba(255,255,255,0.7); font-size: 12px;'>
            Built with ❤️ using Streamlit & AI<br>
            ⚖️ LegalMind AI v2.0
        </div>
        """, unsafe_allow_html=True)

    # Main content layout
    col1, col2 = st.columns([2, 1])

    with col1:
        show_left_panel_content(selected_tab)

    with col2:
        show_right_panel_content(selected_tab)

if __name__ == "__main__":
    main()