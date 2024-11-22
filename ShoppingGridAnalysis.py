import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
import string
import ssl
from io import BytesIO
import zipfile
from wordcloud import WordCloud  # Ensure this is installed: pip install wordcloud

# ***Function to bypass SSL verification for NLTK downloads (Temporary Workaround)***
def nltk_download_ignore_ssl(resource):
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download(resource)

# ***Function to categorize delivery options***
def categorize_delivery(delivery):
    if pd.isna(delivery) or delivery.strip() == '':
        return 'n/a'
    
    delivery = delivery.lower()
    
    # Check if 'free' is present
    if 'free' in delivery:
        # Attempt to find a dollar amount
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

# ***Function: Tokenization and Frequency Analysis***
def tokenize_and_analyze(data):
    # Bypass SSL verification for NLTK downloads (Temporary Workaround)
    nltk_download_ignore_ssl('punkt')
    nltk_download_ignore_ssl('stopwords')
    nltk_download_ignore_ssl('wordnet')
    nltk_download_ignore_ssl('omw-1.4')  # For lemmatizer
    nltk_download_ignore_ssl('averaged_perceptron_tagger')  # For POS tagging

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    punctuations = set(string.punctuation)

    def preprocess(text):
        # Tokenize the text into words
        tokens = word_tokenize(text.lower())
        
        # Remove punctuation and stopwords, and lemmatize the tokens
        cleaned_tokens = [
            lemmatizer.lemmatize(token)
            for token in tokens
            if token not in stop_words and token not in punctuations and token.isalpha()
        ]
        
        return cleaned_tokens

    # Extract all Product_Title columns
    title_columns = [f'Product{i}_Title' for i in range(1, 11)]
    all_titles = data[title_columns].dropna().astype(str).values.flatten()

    # Apply preprocessing to all product titles
    processed_titles = [preprocess(title) for title in all_titles]

    # Flatten the list of tokens
    all_tokens = [token for sublist in processed_titles for token in sublist]

    # Frequency distribution of words
    freq_dist = FreqDist(all_tokens)

    # Convert frequency distribution to DataFrame
    freq_df = pd.DataFrame(freq_dist.items(), columns=['Word', 'Frequency'])
    freq_df = freq_df.sort_values(by='Frequency', ascending=False)

    return freq_dist, freq_df

# ***Function to Generate and Collect Plots***
def analyze_shopping_data(data):
    # Initialize a list to store plot images
    plot_images = []

    # 1. Frequency of each product appearing as the number 1 (top) product
    top_product_counts = data['Product1_Title'].value_counts()
    
    # 2. Price distribution for products in each position (Product1 to Product10)
    price_columns = [f'Product{i}_Price' for i in range(1, 11)]
    price_data = data[price_columns]
    
    # 3. Count of merchants across all products
    merchant_columns = [f'Product{i}_Merchant' for i in range(1, 11)]
    merchant_data = data[merchant_columns].melt(value_name='Merchant').dropna()
    merchant_counts = merchant_data['Merchant'].value_counts()
    
    # 4. Delivery option prevalence analysis with four categories
    delivery_columns = [f'Product{i}_Delivery' for i in range(1, 11)]
    delivery_data = data[delivery_columns].fillna('').astype(str)
    
    # Replaced applymap with apply and map to avoid FutureWarning
    delivery_data_categorized = delivery_data.apply(lambda col: col.map(categorize_delivery))
    
    # Melt the categorized delivery data
    delivery_melted = delivery_data_categorized.melt(value_name='Delivery_Category').dropna()
    
    # Count the categories
    delivery_counts = delivery_melted['Delivery_Category'].value_counts()

    # ***Tokenization and Frequency Analysis***
    freq_dist, freq_df = tokenize_and_analyze(data)

    # Plot 1: Frequency of products appearing as number 1
    st.subheader("Top 10 Products by Frequency in Number 1 Position")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    top_product_counts.head(10).plot(kind='bar', ax=ax1, color='orange')
    ax1.set_xlabel("Product Title")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Top 10 Products by Frequency in Number 1 Position")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig1)
    # Save plot to buffer
    buf1 = BytesIO()
    fig1.savefig(buf1, format="png", bbox_inches='tight')
    plot_images.append(("Top_10_Products.png", buf1.getvalue()))
    plt.close(fig1)

    # Plot 2: Price distribution for each position (Product1 to Product10)
    st.subheader("Price Distribution by Product Position (1 to 10)")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    price_data.boxplot(ax=ax2)
    ax2.set_xlabel("Product Position")
    ax2.set_ylabel("Price")
    ax2.set_title("Price Distribution by Product Position")
    st.pyplot(fig2)
    # Save plot to buffer
    buf2 = BytesIO()
    fig2.savefig(buf2, format="png", bbox_inches='tight')
    plot_images.append(("Price_Distribution.png", buf2.getvalue()))
    plt.close(fig2)

    # Plot 3: Top merchants by frequency of appearance
    st.subheader("Top 10 Merchants by Frequency of Appearance")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    merchant_counts.head(10).plot(kind='bar', ax=ax3, color='green')
    ax3.set_xlabel("Merchant")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Top 10 Merchants by Frequency of Appearance")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig3)
    # Save plot to buffer
    buf3 = BytesIO()
    fig3.savefig(buf3, format="png", bbox_inches='tight')
    plot_images.append(("Top_10_Merchants.png", buf3.getvalue()))
    plt.close(fig3)

    # Plot 4: Delivery option prevalence with new categories
    st.subheader("Prevalence of Delivery Options")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    delivery_counts.plot(kind='bar', ax=ax4, color='skyblue')
    ax4.set_xlabel("Delivery Category")
    ax4.set_ylabel("Frequency")
    ax4.set_title("Prevalence of Delivery Options")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig4)
    # Save plot to buffer
    buf4 = BytesIO()
    fig4.savefig(buf4, format="png", bbox_inches='tight')
    plot_images.append(("Delivery_Options.png", buf4.getvalue()))
    plt.close(fig4)

    # ***New Plot: Top 20 Most Common Words in Product Titles***
    st.subheader("Top 20 Most Common Words in Product Titles")
    fig5, ax5 = plt.subplots(figsize=(12, 8))
    # Remove 'ax=ax5' as FreqDist.plot() does not accept 'ax'
    freq_dist.plot(20, cumulative=False, title='Top 20 Most Common Words')  # Removed 'ax=ax5'
    ax5.set_xlabel("Words")
    ax5.set_ylabel("Frequency")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig5)
    # Save plot to buffer
    buf5 = BytesIO()
    fig5.savefig(buf5, format="png", bbox_inches='tight')
    plot_images.append(("Top_20_Common_Words.png", buf5.getvalue()))
    plt.close(fig5)

    # ***New Feature: Word Cloud***
    st.subheader("Word Cloud of Product Titles")
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(freq_dist)
    fig_wc, ax_wc = plt.subplots(figsize=(15, 7.5))
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis('off')
    st.pyplot(fig_wc)
    # Save plot to buffer
    buf_wc = BytesIO()
    fig_wc.savefig(buf_wc, format="png", bbox_inches='tight')
    plot_images.append(("Word_Cloud.png", buf_wc.getvalue()))
    plt.close(fig_wc)

    # ***Prepare a summary of key insights***
    summary_text = f"""
    - The top product appears {top_product_counts.head(10).sum()} times across the number 1 position.
    - The price across all product positions ranges from ${price_data.min().min()} to ${price_data.max().max()}.
    - The most frequent merchant appears {merchant_counts.iloc[0]} times.
    - Delivery options are predominantly '{delivery_counts.idxmax()}' with a count of {delivery_counts.max()}.
    - The most common words in product titles include: {', '.join(freq_df['Word'].head(10).tolist())}.
    """

    # ***Display the Summary of Key Insights***
    st.subheader("Summary of Key Insights")
    st.write(summary_text)

    # ***Optional: Display Word Frequency Table***
    st.subheader("Word Frequency in Product Titles")
    st.dataframe(freq_df.head(50))  # Display top 50 words

    # ***Optional: Download Frequency Data as CSV***
    st.subheader("Download Word Frequency Data")
    csv = freq_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Frequency Data as CSV",
        data=csv,
        file_name='word_frequency.csv',
        mime='text/csv',
    )

    # ***New Feature: Export All Graphs as a ZIP File***
    st.subheader("Export All Graphs as ZIP")
    if st.button("Download All Graphs"):
        # Create a ZIP file in memory
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            for filename, data in plot_images:
                zip_file.writestr(filename, data)
        zip_buffer.seek(0)
        st.download_button(
            label="Download Graphs ZIP",
            data=zip_buffer,
            file_name='graphs.zip',
            mime='application/zip',
        )

# ***Streamlit App Initialization***
st.title("Shopping Grid Data Analysis C9 v1.0")

# File uploader for CSV file
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

# ***Process the Uploaded File***
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        
        # Check if necessary columns exist
        required_columns = [f'Product{i}_Title' for i in range(1, 11)] + \
                           [f'Product{i}_Price' for i in range(1, 11)] + \
                           [f'Product{i}_Merchant' for i in range(1, 11)] + \
                           [f'Product{i}_Delivery' for i in range(1, 11)]
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            st.error(f"The following required columns are missing in the uploaded CSV: {', '.join(missing_columns)}")
        else:
            analyze_shopping_data(data)
    except LookupError as e:
        st.error(f"NLTK resource missing: {e}. Please ensure all resources are correctly downloaded.")
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
else:
    st.write("Please upload a CSV file to proceed.")
