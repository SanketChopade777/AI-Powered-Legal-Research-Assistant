import pandas as pd
from typing import List, Dict
import streamlit as st


class FeedbackAnalyzer:
    def __init__(self):
        self.feedback_data = []

    def add_feedback(self, feedback: Dict):
        """Add new feedback to analysis"""
        self.feedback_data.append(feedback)

    def get_stats(self) -> Dict:
        """Get feedback statistics"""
        if not self.feedback_data:
            return {}

        df = pd.DataFrame(self.feedback_data)
        stats = {
            'total_feedback': len(df),
            'average_rating': df['rating'].value_counts().to_dict(),
            'common_comments': df['comment'].value_counts().head(5).to_dict()
        }
        return stats

    def generate_report(self) -> str:
        """Generate feedback report"""
        stats = self.get_stats()
        if not stats:
            return "No feedback data available"

        report = [
            "📊 Feedback Analysis Report",
            "=" * 30,
            f"Total Feedback: {stats['total_feedback']}",
            "\nRating Distribution:"
        ]

        for rating, count in stats['average_rating'].items():
            report.append(f"  {rating}: {count}")

        report.append("\nTop Comments:")
        for comment, count in stats['common_comments'].items():
            report.append(f"  - '{comment}': {count}")

        return "\n".join(report)


# Global feedback analyzer instance
feedback_analyzer = FeedbackAnalyzer()