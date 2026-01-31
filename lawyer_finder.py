import streamlit as st
import pandas as pd
import plotly.express as px
import json


def inject_lawyer_css():
    st.markdown("""
    <style>
    /* Main background - Dark theme */
    .stApp {
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


def load_lawyer_data():
    """Load lawyer data from CSV or JSON"""
    try:
        # Try to load from CSV first
        df = pd.read_csv('utils/lawyer_finder/maharashtra_lawyers_dataset.csv')
    except:
        try:
            # If CSV fails, try JSON
            with open('utils/lawyer_finder/maharashtra_lawyers_dataset.json', 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        except:
            # Create a fallback dataset matching your structure
            st.warning("Using sample data. Please upload your real dataset.")
            df = create_sample_dataset_matching_structure()

    # Debug: Show column names
    # st.write("Dataset columns:", df.columns.tolist())

    return df


def create_sample_dataset_matching_structure():
    """Create a sample dataset matching your CSV structure"""
    data = {
        'sr_no': list(range(1, 51)),
        'district': ['Mumbai', 'Pune', 'Nagpur', 'Thane', 'Nashik', 'Aurangabad', 'Solapur',
                     'Amravati', 'Kolhapur', 'Sangli'] * 5,
        'dlsa_tlsc': ['DLSA', 'TLSC'] * 25,
        'name': [f'Adv. {name}' for name in [
            'Rajesh Kumar Sharma', 'Priya Anil Patel', 'Amit Sunil Kumar',
            'Sneha Rajendra Desai', 'Rohan Prakash Verma', 'Anjali Vivek Singh',
            'Vikram Dilip Mehta', 'Neha Sanjay Joshi', 'Sanjay Ramesh Gupta',
            'Pooja Mahesh Reddy', 'Ashwin Iyer', 'Deepika Nair', 'Karan Malhotra',
            'Shweta Choudhary', 'Rahul Bajaj', 'Meera Kapoor', 'Arjun Reddy',
            'Kavita Srinivasan', 'Varun Thakur', 'Sunita Mishra', 'Rajesh Patil',
            'Suresh Joshi', 'Anita Gavde', 'Mahesh Wagh', 'Smita Kulkarni',
            'Prakash Jadhav', 'Varsha More', 'Nitin Tambe', 'Swati Pawar',
            'Yogesh Bhor', 'Sarika Gaikwad', 'Vishwas Shinde', 'Rahul Deshmukh',
            'Priyanka Kulkarni', 'Anil Bhandari', 'Sonali Thakur', 'Rohit Naik',
            'Deepak Mishra', 'Poonam Singh', 'Vikas Rao', 'Nandini Reddy',
            'Manoj Tiwari', 'Shilpa Sharma', 'Abhishek Verma', 'Kiran Patel',
            'Sanjay Mehta', 'Rina Shah', 'Vijay Kumar', 'Anjali Gupta'
        ]],
        'qualification': ['LL.B', 'B.A.LL.B', 'B.Com LL.B', 'LL.M', 'B.S.L.LL.B'] * 10,
        'phone': [f'98{str(i).zfill(8)}' for i in range(12345678, 12345728)],
        'email': [f'advocate{i}@gmail.com' for i in range(1, 51)],
        'empanelled_on': ['01 April 2023'] * 50,
        'empanelment_expiring': ['31 March 2026'] * 50,
        'gender': ['Male', 'Female'] * 25,
        'experience_years': [5, 8, 12, 3, 15, 7, 10, 6, 9, 4] * 5,
        'specialization': [
                              'Criminal Law', 'Civil Law', 'Family Law', 'Corporate Law',
                              'Property Law', 'Labour Law', 'Tax Law', 'Cyber Law',
                              'Intellectual Property', 'Constitutional Law'
                          ] * 5,
        'hourly_rate_inr': [1500, 3000, 2500, 4000, 1800, 3500, 2000, 2800, 3200, 2200] * 5,
        'rating': [4.2, 4.5, 4.8, 3.9, 4.7, 4.1, 4.9, 4.3, 4.6, 4.0] * 5,
        'languages': [
                         'Marathi, Hindi, English', 'Hindi, English', 'Marathi, Hindi',
                         'English, Hindi', 'Marathi, English', 'Hindi',
                         'Marathi, Hindi, English, Gujarati', 'English',
                         'Marathi, Hindi, English', 'Hindi, English'
                     ] * 5,
        'address': [f'{city}, Maharashtra' for city in [
            'Fort, Mumbai', 'Shivajinagar, Pune', 'Sitabuldi, Nagpur',
            'Ghodbunder Road, Thane', 'College Road, Nashik', 'Jalna Road, Aurangabad',
            'Sadar Bazaar, Solapur', 'Rajapeth, Amravati', 'Shahupuri, Kolhapur',
            'Vishrambag, Sangli'
        ]] * 5,
        'court_practice': ['District Court', 'High Court', 'Session Court', 'Family Court'] * 12 + [
            'District Court'] * 2
    }

    return pd.DataFrame(data)


def render_back_button():
    """Render the back button to return to home"""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("← Back to Home", key="doc_back_btn", use_container_width=True):
            if 'current_page' in st.session_state:
                st.session_state.current_page = 'home'
                st.rerun()


def main():
    inject_lawyer_css()

    # Add back button at the top
    render_back_button()

    # Load data
    df = load_lawyer_data()

    # Map column names for compatibility
    # Your dataset uses 'specialization', code expects 'specialty'
    # Your dataset uses 'hourly_rate_inr', code expects 'hourly_rate'
    # Your dataset uses 'experience_years', code expects 'experience'

    # Create aliases for compatibility
    if 'specialization' in df.columns and 'specialty' not in df.columns:
        df['specialty'] = df['specialization']

    if 'hourly_rate_inr' in df.columns and 'hourly_rate' not in df.columns:
        df['hourly_rate'] = df['hourly_rate_inr']

    if 'experience_years' in df.columns and 'experience' not in df.columns:
        df['experience'] = df['experience_years']

    # Create location field if not exists
    if 'location' not in df.columns:
        if 'address' in df.columns:
            df['location'] = df['address']
        elif 'district' in df.columns:
            df['location'] = df['district'] + ', Maharashtra'

    # Sidebar with filters
    with st.sidebar:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #4361ee 0%, #3a0ca3 100%); 
                   padding: 25px; border-radius: 20px; color: white; text-align: center; margin-bottom: 20px;'>
            <h1>👨‍💼 Lawyer Finder</h1>
            <p style='margin: 0;'>Maharashtra Legal Professionals</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🔍 Search Filters")

        # Practice area filter
        if 'specialty' in df.columns:
            practice_areas = ["All Specialties"] + sorted(df['specialty'].unique().tolist())
            selected_specialty = st.selectbox("Practice Area", practice_areas)
        else:
            selected_specialty = "All Specialties"
            st.info("Specialty data not available")

        # Location filter (districts in Maharashtra)
        if 'location' in df.columns:
            locations = ["All Locations"] + sorted(df['location'].unique().tolist())
            selected_location = st.selectbox("Location", locations)
        else:
            selected_location = "All Locations"
            st.info("Location data not available")

        # Experience filter
        if 'experience' in df.columns:
            min_experience = st.slider("Minimum Experience (years)", 0, int(df['experience'].max()), 0)
        else:
            min_experience = 0
            st.info("Experience data not available")

        # Rating filter
        if 'rating' in df.columns:
            min_rating = st.slider("Minimum Rating", 3.0, 5.0, 3.5, 0.1)
        else:
            min_rating = 3.0
            st.info("Rating data not available")

        # Hourly rate filter in INR
        if 'hourly_rate' in df.columns:
            max_rate = st.slider("Maximum Hourly Rate (₹)",
                                 int(df['hourly_rate'].min()),
                                 int(df['hourly_rate'].max()),
                                 int(df['hourly_rate'].max()))
        else:
            max_rate = 5000
            st.info("Hourly rate data not available")

        # Language filter
        if 'languages' in df.columns:
            all_languages = set()
            for lang_string in df['languages']:
                if isinstance(lang_string, str):
                    languages = [l.strip() for l in lang_string.split(',')]
                    all_languages.update(languages)
            languages = ["All Languages"] + sorted(list(all_languages))
            selected_language = st.selectbox("Language", languages)
        else:
            selected_language = "All Languages"
            st.info("Language data not available")

        # DLSA/TLSC filter
        if 'dlsa_tlsc' in df.columns:
            empanelment_options = ["All", "DLSA", "TLSC"]
            selected_empanelment = st.selectbox("Empanelment Type", empanelment_options)
        else:
            selected_empanelment = "All"
            st.info("Empanelment data not available")

        st.markdown("---")
        st.markdown("### 💡 Quick Tips")
        st.info("""
        - Check lawyer reviews and ratings
        - Verify credentials and experience
        - Schedule consultation calls
        - Discuss fees upfront in INR
        - Confirm language proficiency
        """)

    # Main content
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1 style='color: #4cc9f0;'>👨‍💼 Find Your Legal Expert</h1>
        <p style='color: #a8b2d1;'>Connect with qualified lawyers across Maharashtra districts</p>
    </div>
    """, unsafe_allow_html=True)

    # Search bar
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search_query = st.text_input("🔍 Search lawyers by name, specialty, or keywords...")
    with col2:
        if 'rating' in df.columns and 'experience' in df.columns and 'hourly_rate' in df.columns:
            sort_options = ["Rating", "Experience", "Hourly Rate", "Name"]
        else:
            sort_options = ["Name"]
        sort_by = st.selectbox("Sort by", sort_options)
    with col3:
        sort_order = st.selectbox("Order", ["Descending", "Ascending"])

    # Get and filter lawyers
    lawyers_df = df.copy()

    # Apply filters
    if selected_specialty != "All Specialties" and 'specialty' in lawyers_df.columns:
        lawyers_df = lawyers_df[lawyers_df['specialty'] == selected_specialty]

    if selected_location != "All Locations" and 'location' in lawyers_df.columns:
        lawyers_df = lawyers_df[lawyers_df['location'] == selected_location]

    if 'experience' in lawyers_df.columns:
        lawyers_df = lawyers_df[lawyers_df['experience'] >= min_experience]

    if 'rating' in lawyers_df.columns:
        lawyers_df = lawyers_df[lawyers_df['rating'] >= min_rating]

    if 'hourly_rate' in lawyers_df.columns:
        lawyers_df = lawyers_df[lawyers_df['hourly_rate'] <= max_rate]

    if selected_language != "All Languages" and 'languages' in lawyers_df.columns:
        lawyers_df = lawyers_df[lawyers_df['languages'].str.contains(selected_language, na=False)]

    if selected_empanelment != "All" and 'dlsa_tlsc' in lawyers_df.columns:
        lawyers_df = lawyers_df[lawyers_df['dlsa_tlsc'] == selected_empanelment]

    # Apply search query
    if search_query:
        search_conditions = []
        if 'name' in lawyers_df.columns:
            search_conditions.append(lawyers_df['name'].str.contains(search_query, case=False, na=False))
        if 'specialty' in lawyers_df.columns:
            search_conditions.append(lawyers_df['specialty'].str.contains(search_query, case=False, na=False))
        if 'languages' in lawyers_df.columns:
            search_conditions.append(lawyers_df['languages'].str.contains(search_query, case=False, na=False))
        if 'location' in lawyers_df.columns:
            search_conditions.append(lawyers_df['location'].str.contains(search_query, case=False, na=False))

        if search_conditions:
            combined_condition = search_conditions[0]
            for condition in search_conditions[1:]:
                combined_condition = combined_condition | condition
            lawyers_df = lawyers_df[combined_condition]

    # Sort results
    if sort_by == "Rating" and 'rating' in lawyers_df.columns:
        lawyers_df = lawyers_df.sort_values('rating', ascending=(sort_order == "Ascending"))
    elif sort_by == "Experience" and 'experience' in lawyers_df.columns:
        lawyers_df = lawyers_df.sort_values('experience', ascending=(sort_order == "Ascending"))
    elif sort_by == "Hourly Rate" and 'hourly_rate' in lawyers_df.columns:
        lawyers_df = lawyers_df.sort_values('hourly_rate', ascending=(sort_order == "Ascending"))
    elif 'name' in lawyers_df.columns:
        lawyers_df = lawyers_df.sort_values('name', ascending=(sort_order == "Ascending"))

    # Display results
    st.markdown(f"### 📊 Found {len(lawyers_df)} Qualified Lawyers")

    if len(lawyers_df) > 0:
        for idx, lawyer in lawyers_df.iterrows():
            # Create rating stars
            if 'rating' in lawyer:
                stars = "⭐" * int(lawyer['rating']) + "☆" * (5 - int(lawyer['rating']))
                rating_text = f"{stars} ({lawyer['rating']})"
            else:
                rating_text = "Rating not available"

            # Build specialty text
            specialty_text = ""
            if 'specialty' in lawyer:
                specialty_text += lawyer['specialty']
            if 'location' in lawyer:
                if specialty_text:
                    specialty_text += " • "
                specialty_text += lawyer['location']

            # Build experience text
            experience_text = ""
            if 'experience' in lawyer:
                experience_text = f"{lawyer['experience']} years experience"

            st.markdown(f"""
            <div class='lawyer-card'>
                <div class='lawyer-name'>{lawyer.get('name', 'Name not available')}</div>
                <div class='lawyer-specialty'>{specialty_text}</div>
                <div class='rating'>{rating_text} • {experience_text}</div>
                <p>💼 <strong>Hourly Rate:</strong> ₹{lawyer.get('hourly_rate', 'N/A')}/hr • 🗣️ <strong>Languages:</strong> {lawyer.get('languages', 'Not specified')}</p>
                <p>📞 <strong>Phone:</strong> {lawyer.get('phone', 'Not available')} • ✉️ <strong>Email:</strong> {lawyer.get('email', 'Not available')}</p>
                {'<p>🏛️ <strong>Empanelment:</strong> ' + lawyer['dlsa_tlsc'] + '</p>' if 'dlsa_tlsc' in lawyer else ''}
            </div>
            """, unsafe_allow_html=True)

            # Action buttons
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            with col1:
                if st.button("📞 Call", key=f"call_{idx}", use_container_width=True):
                    st.success(f"Calling {lawyer.get('name', 'Lawyer')}...")
            with col2:
                if st.button("📧 Email", key=f"email_{idx}", use_container_width=True):
                    st.info(f"Emailing {lawyer.get('name', 'Lawyer')}...")
            with col3:
                if st.button("📅 Schedule", key=f"schedule_{idx}", use_container_width=True):
                    st.info(f"Scheduling consultation with {lawyer.get('name', 'Lawyer')}...")
            with col4:
                if st.button("📋 Profile", key=f"profile_{idx}", use_container_width=True):
                    # Show detailed profile
                    with st.expander(f"Full Profile - {lawyer.get('name', 'Lawyer')}"):
                        profile_html = """
                        ### 📋 Complete Profile

                        **Personal Information:**
                        """

                        if 'name' in lawyer:
                            profile_html += f"\n- **Name:** {lawyer['name']}"
                        if 'gender' in lawyer:
                            profile_html += f"\n- **Gender:** {lawyer['gender']}"
                        if 'location' in lawyer:
                            profile_html += f"\n- **Location:** {lawyer['location']}"
                        if 'address' in lawyer:
                            profile_html += f"\n- **Address:** {lawyer['address']}"

                        profile_html += "\n\n**Professional Information:**"

                        if 'specialty' in lawyer:
                            profile_html += f"\n- **Specialty:** {lawyer['specialty']}"
                        if 'qualification' in lawyer:
                            profile_html += f"\n- **Qualification:** {lawyer['qualification']}"
                        if 'experience' in lawyer:
                            profile_html += f"\n- **Experience:** {lawyer['experience']} years"
                        if 'rating' in lawyer:
                            profile_html += f"\n- **Rating:** {lawyer['rating']}/5.0"
                        if 'court_practice' in lawyer:
                            profile_html += f"\n- **Court Practice:** {lawyer['court_practice']}"

                        profile_html += "\n\n**Contact Information:**"

                        if 'phone' in lawyer:
                            profile_html += f"\n- **Phone:** {lawyer['phone']}"
                        if 'email' in lawyer:
                            profile_html += f"\n- **Email:** {lawyer['email']}"
                        if 'hourly_rate' in lawyer:
                            profile_html += f"\n- **Hourly Rate:** ₹{lawyer['hourly_rate']}"

                        profile_html += "\n\n**Additional Information:**"

                        if 'languages' in lawyer:
                            profile_html += f"\n- **Languages:** {lawyer['languages']}"
                        if 'dlsa_tlsc' in lawyer:
                            profile_html += f"\n- **Empanelment:** {lawyer['dlsa_tlsc']}"
                        if 'empanelled_on' in lawyer:
                            profile_html += f"\n- **Empanelment Date:** {lawyer['empanelled_on']}"
                        if 'empanelment_expiring' in lawyer:
                            profile_html += f"\n- **Empanelment Expiry:** {lawyer['empanelment_expiring']}"

                        st.markdown(profile_html)

            st.markdown("---")
    else:
        st.warning("No lawyers found matching your criteria. Try adjusting your filters.")

    # Statistics section
    if len(lawyers_df) > 0:
        st.markdown("### 📈 Market Insights")
        cols = st.columns(4)

        col_index = 0

        if 'hourly_rate' in lawyers_df.columns:
            with cols[col_index]:
                avg_rate = lawyers_df['hourly_rate'].mean()
                st.metric("💵 Average Hourly Rate", f"₹{avg_rate:.0f}")
            col_index += 1

        if 'experience' in lawyers_df.columns:
            with cols[col_index]:
                avg_exp = lawyers_df['experience'].mean()
                st.metric("🎓 Average Experience", f"{avg_exp:.1f} years")
            col_index += 1

        if 'rating' in lawyers_df.columns:
            with cols[col_index]:
                avg_rating = lawyers_df['rating'].mean()
                st.metric("⭐ Average Rating", f"{avg_rating:.1f}/5.0")
            col_index += 1

        if 'location' in lawyers_df.columns:
            with cols[col_index]:
                unique_locations = lawyers_df['location'].nunique()
                st.metric("📍 Districts", f"{unique_locations}")

        # Visualizations (only show if we have data)
        st.markdown("---")
        st.markdown("### 📊 Distribution Analysis")

        visualization_tabs = []

        if 'location' in lawyers_df.columns and len(lawyers_df) > 0:
            visualization_tabs.append("By District")

        if 'specialty' in lawyers_df.columns and len(lawyers_df) > 0:
            visualization_tabs.append("By Specialty")

        if 'hourly_rate' in lawyers_df.columns and 'specialty' in lawyers_df.columns and len(lawyers_df) > 0:
            visualization_tabs.append("Fee Analysis")

        if visualization_tabs:
            tabs = st.tabs(visualization_tabs)

            tab_index = 0

            if 'location' in lawyers_df.columns and len(lawyers_df) > 0:
                with tabs[tab_index]:
                    location_counts = lawyers_df['location'].value_counts().reset_index()
                    location_counts.columns = ['Location', 'Count']
                    fig1 = px.bar(location_counts.head(10),
                                  x='Location',
                                  y='Count',
                                  title="Top 10 Locations by Lawyer Count",
                                  color='Count',
                                  color_continuous_scale='viridis')
                    st.plotly_chart(fig1, use_container_width=True)
                tab_index += 1

            if 'specialty' in lawyers_df.columns and len(lawyers_df) > 0:
                with tabs[tab_index]:
                    specialty_counts = lawyers_df['specialty'].value_counts().reset_index()
                    specialty_counts.columns = ['Specialty', 'Count']
                    fig2 = px.pie(specialty_counts,
                                  values='Count',
                                  names='Specialty',
                                  title="Distribution by Legal Specialty",
                                  hole=0.3)
                    st.plotly_chart(fig2, use_container_width=True)
                tab_index += 1

            if 'hourly_rate' in lawyers_df.columns and 'specialty' in lawyers_df.columns and len(lawyers_df) > 0:
                with tabs[tab_index]:
                    fig3 = px.box(lawyers_df,
                                  x='specialty',
                                  y='hourly_rate',
                                  title="Hourly Rate Distribution by Specialty",
                                  labels={'hourly_rate': 'Hourly Rate (₹)', 'specialty': 'Specialty'})
                    st.plotly_chart(fig3, use_container_width=True)

        # Download option
        st.markdown("---")
        st.markdown("### 📥 Download Results")

        col1, col2 = st.columns(2)
        with col1:
            csv = lawyers_df.to_csv(index=False)
            st.download_button(
                label="📄 Download CSV",
                data=csv,
                file_name="maharashtra_lawyers.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col2:
            json_data = lawyers_df.to_json(orient='records', indent=2)
            st.download_button(
                label="📁 Download JSON",
                data=json_data,
                file_name="maharashtra_lawyers.json",
                mime="application/json",
                use_container_width=True
            )


if __name__ == "__main__":
    main()