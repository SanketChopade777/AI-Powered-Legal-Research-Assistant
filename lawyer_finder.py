import streamlit as st
import pandas as pd
import plotly.express as px
import webbrowser


def inject_lawyer_css():
    st.markdown("""
    <style>
    /* Main background - Dark theme */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #ffffff;
    }

    /* Lawyer cards */
    .lawyer-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        border-left: 5px solid #4cc9f0;
        transition: all 0.3s ease;
    }

    .lawyer-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(76, 201, 240, 0.3);
        border-left-color: #f72585;
    }

    .lawyer-name {
        color: #4cc9f0;
        font-size: 1.4rem;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .lawyer-specialty {
        color: #f72585;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .rating {
        color: #ffd700;
        font-size: 1.1rem;
    }
    </style>
    """, unsafe_allow_html=True)


# Sample lawyer data
def get_sample_lawyers():
    return pd.DataFrame({
        'name': ['Dr. Sarah Johnson', 'Robert Chen', 'Maria Rodriguez', 'James Wilson',
                 'Priya Patel', 'Michael Brown', 'Emily Davis', 'David Kim'],
        'specialty': ['Corporate Law', 'Criminal Defense', 'Family Law', 'Intellectual Property',
                      'Immigration Law', 'Personal Injury', 'Real Estate', 'Tax Law'],
        'experience': [15, 8, 12, 10, 7, 9, 11, 14],
        'rating': [4.8, 4.6, 4.9, 4.7, 4.5, 4.8, 4.6, 4.9],
        'location': ['New York, NY', 'Los Angeles, CA', 'Chicago, IL', 'San Francisco, CA',
                     'Houston, TX', 'Miami, FL', 'Boston, MA', 'Seattle, WA'],
        'hourly_rate': [350, 275, 300, 400, 250, 325, 280, 375],
        'languages': ['English, Spanish', 'English, Mandarin', 'English, Spanish', 'English',
                      'English, Hindi, Gujarati', 'English', 'English, French', 'English, Korean'],
        'availability': ['Immediate', 'Next Week', 'Immediate', '2 Weeks',
                         'Immediate', 'Next Week', 'Immediate', '3 Days']
    })


def main():
    # st.set_page_config(
    #     page_title="LegalEase AI - Lawyer Finder",
    #     layout="wide",
    #     page_icon="👨‍💼",
    #     initial_sidebar_state="expanded"
    # )

    inject_lawyer_css()

    # Sidebar with filters
    with st.sidebar:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #4361ee 0%, #3a0ca3 100%); 
                   padding: 25px; border-radius: 20px; color: white; text-align: center; margin-bottom: 20px;'>
            <h1>👨‍💼 Lawyer Finder</h1>
            <p style='margin: 0;'>Find Your Perfect Legal Match</p>
        </div>
        """, unsafe_allow_html=True)

        # if st.button("🏠 Back to Main Menu", use_container_width=True):
        #     webbrowser.open_new_tab("http://localhost:8501")

        st.markdown("---")
        st.markdown("### 🔍 Search Filters")

        # Practice area filter
        practice_areas = [
            "All Specialties", "Corporate Law", "Criminal Defense", "Family Law",
            "Intellectual Property", "Immigration Law", "Personal Injury",
            "Real Estate", "Tax Law", "Employment Law"
        ]
        selected_specialty = st.selectbox("Practice Area", practice_areas)

        # Location filter
        locations = ["All Locations", "New York, NY", "Los Angeles, CA", "Chicago, IL",
                     "San Francisco, CA", "Houston, TX", "Miami, FL", "Boston, MA", "Seattle, WA"]
        selected_location = st.selectbox("Location", locations)

        # Experience filter
        min_experience = st.slider("Minimum Experience (years)", 0, 20, 5)

        # Rating filter
        min_rating = st.slider("Minimum Rating", 3.0, 5.0, 4.0, 0.1)

        # Hourly rate filter
        max_rate = st.slider("Maximum Hourly Rate ($)", 100, 500, 350)

        st.markdown("---")
        st.markdown("### 💡 Quick Tips")
        st.info("""
        - Check lawyer reviews and ratings
        - Verify credentials and experience
        - Schedule consultation calls
        - Discuss fees upfront
        """)

    # Main content
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1 style='color: #4cc9f0;'>👨‍💼 Find Your Legal Expert</h1>
        <p style='color: #a8b2d1;'>Connect with qualified lawyers tailored to your specific needs and preferences</p>
    </div>
    """, unsafe_allow_html=True)

    # Search bar
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search_query = st.text_input("🔍 Search lawyers by name, specialty, or keywords...")
    with col2:
        sort_by = st.selectbox("Sort by", ["Rating", "Experience", "Hourly Rate"])
    with col3:
        sort_order = st.selectbox("Order", ["Descending", "Ascending"])

    # Get and filter lawyers
    lawyers_df = get_sample_lawyers()

    # Apply filters
    if selected_specialty != "All Specialties":
        lawyers_df = lawyers_df[lawyers_df['specialty'] == selected_specialty]

    if selected_location != "All Locations":
        lawyers_df = lawyers_df[lawyers_df['location'] == selected_location]

    lawyers_df = lawyers_df[lawyers_df['experience'] >= min_experience]
    lawyers_df = lawyers_df[lawyers_df['rating'] >= min_rating]
    lawyers_df = lawyers_df[lawyers_df['hourly_rate'] <= max_rate]

    # Apply search query
    if search_query:
        lawyers_df = lawyers_df[
            lawyers_df['name'].str.contains(search_query, case=False) |
            lawyers_df['specialty'].str.contains(search_query, case=False) |
            lawyers_df['languages'].str.contains(search_query, case=False)
            ]

    # Sort results
    if sort_by == "Rating":
        lawyers_df = lawyers_df.sort_values('rating', ascending=(sort_order == "Ascending"))
    elif sort_by == "Experience":
        lawyers_df = lawyers_df.sort_values('experience', ascending=(sort_order == "Ascending"))
    else:
        lawyers_df = lawyers_df.sort_values('hourly_rate', ascending=(sort_order == "Ascending"))

    # Display results
    st.markdown(f"### 📊 Found {len(lawyers_df)} Qualified Lawyers")

    if len(lawyers_df) > 0:
        for idx, lawyer in lawyers_df.iterrows():
            # Create rating stars
            stars = "⭐" * int(lawyer['rating']) + "☆" * (5 - int(lawyer['rating']))

            st.markdown(f"""
            <div class='lawyer-card'>
                <div class='lawyer-name'>{lawyer['name']}</div>
                <div class='lawyer-specialty'>{lawyer['specialty']} • {lawyer['location']}</div>
                <div class='rating'>{stars} ({lawyer['rating']}) • {lawyer['experience']} years experience</div>
                <p>💼 <strong>Hourly Rate:</strong> ${lawyer['hourly_rate']}/hr • 🗣️ <strong>Languages:</strong> {lawyer['languages']}</p>
                <p>📅 <strong>Availability:</strong> {lawyer['availability']}</p>
            </div>
            """, unsafe_allow_html=True)

            # Action buttons
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button("📞 Contact", key=f"contact_{idx}", use_container_width=True):
                    st.success(f"Contacting {lawyer['name']}...")
            with col2:
                if st.button("💼 Schedule", key=f"schedule_{idx}", use_container_width=True):
                    st.info(f"Scheduling consultation with {lawyer['name']}...")
            with col3:
                if st.button("📋 View Profile", key=f"profile_{idx}", use_container_width=True):
                    st.session_state.selected_lawyer = lawyer['name']
                    st.rerun()

            st.markdown("---")
    else:
        st.warning("No lawyers found matching your criteria. Try adjusting your filters.")

    # Statistics section
    if len(lawyers_df) > 0:
        st.markdown("### 📈 Market Insights")
        col1, col2, col3 = st.columns(3)

        with col1:
            avg_rate = lawyers_df['hourly_rate'].mean()
            st.metric("💵 Average Hourly Rate", f"${avg_rate:.0f}")

        with col2:
            avg_exp = lawyers_df['experience'].mean()
            st.metric("🎓 Average Experience", f"{avg_exp:.1f} years")

        with col3:
            avg_rating = lawyers_df['rating'].mean()
            st.metric("⭐ Average Rating", f"{avg_rating:.1f}/5.0")


if __name__ == "__main__":
    main()