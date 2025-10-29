def inject_navigation_css():
    return """
    <style>
    /* Main background - Dark theme */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Navigation cards with enhanced animations */
    .nav-card {
        background: rgba(26, 26, 46, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 40px 30px;
        margin: 20px 0;
        box-shadow: 0 15px 35px rgba(0,0,0,0.5);
        border: 2px solid rgba(76, 201, 240, 0.3);
        transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        text-align: center;
        color: white;
        height: 320px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        position: relative;
        overflow: hidden;
        opacity: 0;
        transform: translateY(30px);
        animation: fadeInUp 0.8s ease forwards;
    }

    .nav-card:nth-child(1) { animation-delay: 0.2s; }
    .nav-card:nth-child(2) { animation-delay: 0.4s; }
    .nav-card:nth-child(3) { animation-delay: 0.6s; }

    @keyframes fadeInUp {
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .nav-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(76, 201, 240, 0.2), transparent);
        transition: left 0.6s;
    }

    .nav-card:hover::before {
        left: 100%;
    }

    .nav-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 25px 50px rgba(76, 201, 240, 0.3);
        border-color: #f72585;
    }

    .nav-icon {
        font-size: 4rem;
        margin-bottom: 20px;
        background: linear-gradient(135deg, #4cc9f0 0%, #f72585 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 5px 15px rgba(76, 201, 240, 0.5));
        transition: all 0.3s ease;
    }

    .nav-card:hover .nav-icon {
        transform: scale(1.1) rotate(5deg);
        filter: drop-shadow(0 8px 25px rgba(247, 37, 133, 0.6));
    }

    .nav-title {
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 15px;
        background: linear-gradient(135deg, #94d2e5 0%, #ffffff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .nav-description {
        font-size: 1rem;
        opacity: 0.9;
        margin-bottom: 20px;
        line-height: 1.5;
    }

    .stButton>button {
        background: linear-gradient(135deg, #f72585 0%, #b5179e 100%);
        color: white;
        border: none;
        padding: 12px 10px;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 40%;
    }

    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 25px rgba(247, 37, 133, 0.4);
        background: linear-gradient(135deg, #b5179e 0%, #7209b7 60%);
    }

    /* Header styling */
    .main-header {
        text-align: center;
        padding: 40px 20px;
        margin-bottom: 30px;
    }

    .main-title {
        font-size: 4rem;
        font-weight: bold;
        color: #4cc9f0;
        margin-bottom: 15px;
    }

    .main-subtitle {
        font-size: 1.4rem;
        color: #a8b2d1;
        max-width: 600px;
        margin: 0 auto;
    }

    /* Footer styling */
    .footer {
        text-align: center;
        color: #a8b2d1;
        padding: 40px 20px;
        margin-top: 60px;
        border-top: 1px solid rgba(76, 201, 240, 0.3);
        background: rgba(10, 10, 20, 0.5);
        backdrop-filter: blur(10px);
    }

     /* Loading spinner */
    .loading-spinner {
        display: inline-block;
        width: 50px;
        height: 50px;
        border: 5px solid rgba(76, 201, 240, 0.3);
        border-radius: 50%;
        border-top-color: #4cc9f0;
        animation: spin 1s ease-in-out infinite;
        margin: 20px auto;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    /* Feature cards animation */
    .feature-card {
        background: rgba(26, 26, 46, 0.6);
        border-radius: 15px;
        padding: 30px;
        margin: 15px 0;
        border-left: 4px solid #4cc9f0;
        transition: all 0.3s ease;
        opacity: 0;
        transform: translateY(30px);
        animation: fadeInUp 0.8s ease forwards;
    }

    .feature-card:nth-child(1) { animation-delay: 1s; }
    .feature-card:nth-child(2) { animation-delay: 1.2s; }
    .feature-card:nth-child(3) { animation-delay: 1.4s; }

    .feature-card:hover {
        background: rgba(26, 26, 46, 0.9);
        transform: translateY(-5px);
        border-left-color: #f72585;
    }
    </style>
    """