import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
import io
import zipfile
import os
from dotenv import load_dotenv

# Import the new OpenAI client style
from openai import OpenAI

# --------------------------------------
# Load environment variables from .env
# --------------------------------------
load_dotenv()  # Reads .env file and loads into os.environ
api_key = os.environ.get("OPENAI_API_KEY")

# Create a client instance using the new approach
client = OpenAI(api_key=api_key)

# --------------------------------------
# Streamlit App Title
# --------------------------------------
st.title("Popular Products Analysis")

# --------------------------------------
# File Uploader for CSV
# --------------------------------------
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

# --------------------------------------
# 1) HELPER FUNCTIONS
# --------------------------------------

def categorize_delivery(delivery):
    """Categorize delivery options into a few buckets."""
    import pandas as pd
    if pd.isna(delivery) or delivery.strip() == '':
        return 'n/a'
    
    delivery = delivery.lower()
    
    # Check for 'free'
    if 'free' in delivery:
        # Check for $amount
        import re
        match = re.search(r'\$([0-9]+)', delivery)
        if match:
            amount = int(match.group(1))
            if amount >= 50:
                return 'Free Shipping $50 and Over'
            else:
                return 'Free Shipping $50 and Under'
        else:
            return 'Free Shipping'
    else:
        return 'No Free Shipping'


def generate_ai_insights(prompt, model="gpt-4", temperature=0.5):
    """
    Send a chat completion request to OpenAI using the updated client-based approach.
    """
    if not api_key:
        return "**No OPENAI_API_KEY found.** Please check your .env file."
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=1500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred while generating insights: {e}"


def analyze_shopping_data(data):
    """
    Perform basic analysis and plotting, 
    returning plots as (name, bytes) for ZIP export,
    and returning textual summaries for AI consumption.
    """
    plot_buffers = []  # list of (filename, binary_data)

    # 1) Frequency of each product in #1 position
    top_product_counts = data['Product1_Title'].value_counts()

    # 2) Price distribution
    price_columns = [f'Product{i}_Price' for i in range(1, 11)]
    price_data = data[price_columns]

    # 3) Merchant frequencies
    merchant_columns = [f'Product{i}_Merchant' for i in range(1, 11)]
    merchant_data = data[merchant_columns].melt(value_name='Merchant').dropna()
    merchant_counts = merchant_data['Merchant'].value_counts()

    # 4) Delivery categories
    delivery_columns = [f'Product{i}_Delivery' for i in range(1, 11)]
    delivery_data = data[delivery_columns].fillna('').astype(str)
    delivery_data_categorized = delivery_data.apply(lambda col: col.map(categorize_delivery))
    delivery_melted = delivery_data_categorized.melt(value_name='Delivery_Category').dropna()
    delivery_counts = delivery_melted['Delivery_Category'].value_counts()

    # --- PLOT 1: Top 10 Products in #1 position ---
    st.subheader("Top 10 Products by Frequency in #1 Position")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    top_product_counts.head(10).plot(kind='bar', ax=ax1, color='orange')
    ax1.set_title("Top 10 Products in #1 Position")
    ax1.set_xlabel("Product Title")
    ax1.set_ylabel("Frequency")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig1)
    # Save to buffer
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format="png", bbox_inches='tight')
    plot_buffers.append(("top_10_products.png", buf1.getvalue()))
    plt.close(fig1)

    # --- PLOT 2: Price Distribution ---
    st.subheader("Price Distribution by Product Position")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    price_data.boxplot(ax=ax2)
    ax2.set_title("Price Distribution (Positions 1-10)")
    ax2.set_xlabel("Product Position")
    ax2.set_ylabel("Price")
    st.pyplot(fig2)
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format="png", bbox_inches='tight')
    plot_buffers.append(("price_distribution.png", buf2.getvalue()))
    plt.close(fig2)

    # --- PLOT 3: Top 10 Merchants ---
    st.subheader("Top 10 Merchants by Frequency")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    merchant_counts.head(10).plot(kind='bar', ax=ax3, color='green')
    ax3.set_title("Top 10 Merchants")
    ax3.set_xlabel("Merchant")
    ax3.set_ylabel("Frequency")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig3)
    buf3 = io.BytesIO()
    fig3.savefig(buf3, format="png", bbox_inches='tight')
    plot_buffers.append(("top_10_merchants.png", buf3.getvalue()))
    plt.close(fig3)

    # --- PLOT 4: Delivery Option Prevalence ---
    st.subheader("Prevalence of Delivery Options")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    delivery_counts.plot(kind='bar', ax=ax4, color='skyblue')
    ax4.set_title("Delivery Options")
    ax4.set_xlabel("Delivery Category")
    ax4.set_ylabel("Frequency")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig4)
    buf4 = io.BytesIO()
    fig4.savefig(buf4, format="png", bbox_inches='tight')
    plot_buffers.append(("delivery_options.png", buf4.getvalue()))
    plt.close(fig4)

    # --- BASIC SUMMARY TEXT (for AI and display) ---
    summary_text = f"""
    **Basic Data Insights**:
    - Top 10 products in #1 position (frequency): 
      {top_product_counts.head(10).to_dict()}
    - Price range: from {price_data.min().min()} to {price_data.max().max()}.
    - Top 10 merchants (frequency): 
      {merchant_counts.head(10).to_dict()}
    - Delivery categories count: 
      {delivery_counts.to_dict()}
    """

    return plot_buffers, summary_text


def export_all(plot_buffers, original_csv):
    """Allow user to download all plots + original CSV in a single ZIP."""
    with zipfile.ZipFile("all_exports.zip", "w") as zf:
        # Add each plot
        for filename, data in plot_buffers:
            zf.writestr(filename, data)
        # Add the original CSV
        zf.writestr("original_data.csv", original_csv.getvalue())
    with open("all_exports.zip", "rb") as f:
        st.download_button(
            label="Download All (ZIP)",
            data=f.read(),
            file_name="all_exports.zip",
            mime="application/zip"
        )

# --------------------------------------
# 2) MAIN APP LOGIC
# --------------------------------------
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        
        # Check required columns exist
        required_cols = []
        for i in range(1, 11):
            required_cols.extend([
                f'Product{i}_Title',
                f'Product{i}_Price',
                f'Product{i}_Merchant',
                f'Product{i}_Delivery'
            ])
        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            # --- Run Analysis & Get Plots ---
            plot_buffers, summary_text = analyze_shopping_data(data)

            # --- Export Everything Option ---
            st.subheader("Export All Outputs")
            st.markdown("Download all generated plots plus the original CSV in one ZIP file.")
            export_all_button = st.button("Create & Download ZIP")
            if export_all_button:
                export_all(plot_buffers, uploaded_file)

            # Instead of passing the entire data, let's pass only summary_text 
            # (and optionally, a small sample of the data if you want)
            small_sample = data.head(5).to_dict(orient="records")
            # Convert that small sample to JSON for clarity, or leave it out entirely
            # You can do:
            # sample_json = json.dumps(small_sample, indent=2)

            default_prompt = f"""
            You have access to the following dataset insights. 
            Provide insights and recommendations for Google Merchant Center strategy.
            Reference specific product or merchant details from the summary if needed.

            DATA SUMMARY:
            {summary_text}

            # Optionally, a small sample of raw rows
            (First 5 rows):
            {small_sample}

            ###
            """

            st.subheader("AI-Generated Insights")
            with st.spinner("Doing AI shit"):
                # Now using the updated client approach with 'generate_ai_insights'
                ai_response = generate_ai_insights(default_prompt, model="gpt-4", temperature=0.2)
            st.write(ai_response)

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a CSV file to begin. Make sure OPENAI_API_KEY is in your .env.")

st.markdown("---")
st.markdown("© 2025 Calibre Nine | [GitHub Repository](https://github.com/chrisprideC9)")   