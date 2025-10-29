from utils.document_upload.document_analysis_component import DocumentAnalysisComponent


def main():
    # No set_page_config() here - it's already called in the main app

    # Initialize the document analysis component
    doc_analyzer = DocumentAnalysisComponent()

    # Run the main analysis interface
    doc_analyzer.main()


if __name__ == "__main__":
    main()