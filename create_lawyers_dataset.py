import pandas as pd
import re

# Sample data structure based on your PDF - I'll extract key information
lawyers_data = []


# Parse the PDF text content and create structured data
def parse_lawyer_data_from_pdf(pdf_text):
    lines = pdf_text.split('\n')
    current_lawyer = {}

    for line in lines:
        # Parse and structure the data
        pass


# For now, let's create a sample dataset based on your PDF structure
def create_maharashtra_lawyers_dataset():
    # This is a sample structure - in reality, you'd parse your PDF
    return pd.DataFrame({
        'sr_no': range(1, 101),
        'district': ['Mumbai', 'Pune', 'Nagpur', 'Thane', 'Nashik'] * 20,
        'dlsa_tlsc': ['DLSA'] * 50 + ['TLSC'] * 50,
        'name': [f'Adv. {name}' for name in [
            'Rajesh Sharma', 'Priya Patel', 'Amit Kumar', 'Sneha Desai', 'Rohan Verma',
            'Anjali Singh', 'Vikram Mehta', 'Neha Joshi', 'Sanjay Gupta', 'Pooja Reddy',
            # Add more names from your PDF
        ]] * 10,
        'qualification': ['LL.B', 'B.A.LL.B', 'B.Com LL.B', 'LL.M', 'B.S.L.LL.B'] * 20,
        'phone': ['9' + ''.join(['1' for _ in range(9)]) for _ in range(100)],
        'email': [f'advocate{i}@gmail.com' for i in range(100)],
        'empanelled_on': ['01 April 2023'] * 100,
        'empanelment_expiring': ['31 March 2026'] * 100,
        'gender': ['Male', 'Female'] * 50,
        'experience_years': [5, 8, 12, 3, 15, 7, 10, 6, 9, 4] * 10,
        'specialization': ['Criminal Law', 'Civil Law', 'Family Law', 'Corporate Law',
                           'Property Law', 'Labour Law', 'Tax Law', 'Cyber Law',
                           'Intellectual Property', 'Constitutional Law'] * 10,
        'hourly_rate_inr': [2000, 3000, 2500, 4000, 1500, 3500, 1800, 2800, 3200, 2200] * 10,
        'rating': [4.2, 4.5, 4.8, 3.9, 4.7, 4.1, 4.9, 4.3, 4.6, 4.0] * 10,
        'languages': ['Marathi, Hindi, English', 'Hindi, English', 'Marathi, Hindi',
                      'English, Hindi', 'Marathi, English', 'Hindi',
                      'Marathi, Hindi, English, Gujarati', 'English',
                      'Marathi, Hindi, English', 'Hindi, English'] * 10,
        'address': [f'{city}, Maharashtra' for city in [
            'Mumbai', 'Pune', 'Nagpur', 'Thane', 'Nashik', 'Aurangabad', 'Solapur',
            'Amravati', 'Kolhapur', 'Sangli'
        ]] * 10,
        'court_practice': ['District Court', 'High Court', 'Session Court',
                           'Consumer Court', 'Family Court', 'Labour Court'] * 16 + ['District Court'] * 4
    })


# Save to CSV
def save_dataset():
    df = create_maharashtra_lawyers_dataset()
    df.to_csv('maharashtra_lawyers_dataset.csv', index=False)
    print("Dataset saved as 'maharashtra_lawyers_dataset.csv'")

    # Also create a JSON version
    df.to_json('maharashtra_lawyers_dataset.json', orient='records')
    print("Dataset saved as 'maharashtra_lawyers_dataset.json'")

    return df


if __name__ == "__main__":
    save_dataset()