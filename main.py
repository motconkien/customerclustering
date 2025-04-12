import streamlit as st
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import pickle
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from itertools import combinations

#-----------------PREPARE-----------------#
product_data = pd.read_csv("./data/Products_with_Categories.csv")
transaction_data = pd.read_csv('./data/Transactions.csv')
RFM_data = pd.read_csv('./data/df_RFM.csv')
raw_data = pd.read_csv('./data/rawData.csv')
product_count = product_data["productName"].nunique()
category_count = product_data["Category"].nunique()
member_count = transaction_data['Member_number'].nunique()

# Merge to get price per transaction
merged = transaction_data.merge(product_data, on='productId')
merged['revenue'] = merged['items'] * merged['price']

# Sort products by price
sorted_products = product_data.sort_values(by='price')

# Get top 5 most expensive and cheapest products
top_expensive = sorted_products.tail(5)
top_cheap = sorted_products.head(5)

# Revenue and quantity sold for each
def get_product_revenue_summary(products_df, label):
    selected = merged[merged['productId'].isin(products_df['productId'])]
    summary = selected.groupby('productId').agg(
        productName=('productName', 'first'),
        category=('Category', 'first'),
        price=('price', 'first'),
        quantity_sold=('items', 'sum'),
        total_revenue=('revenue', 'sum')
    ).reset_index()
    summary['type'] = label
    return summary

expensive_summary = get_product_revenue_summary(top_expensive, 'Most Expensive')
cheap_summary = get_product_revenue_summary(top_cheap, 'Cheapest')

# Combine
compare_df = pd.concat([expensive_summary, cheap_summary])

#load all models and scaler 
with open('./models/kmeans_model.pkl', 'rb') as f:
    kmeans_model = pickle.load(f)

with open('./models/gmm_model.pkl', 'rb') as f:
    gmm_model = pickle.load(f)

with open('./models/kmeans_model_3.pkl', 'rb') as f:
    kmeans_model_3 = pickle.load(f)

with open('./models/gmm_model_3.pkl', 'rb') as f:
    gmm_model_3 = pickle.load(f)

with open('./models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

#prepare character to display, has to use df_RFM
df_RFM = RFM_data[['Recency', 'Frequency', 'Monetary']]
df_RFM_3 = df_RFM.copy()
df_RFM_3['cluster'] = kmeans_model_3.labels_
df_RFM_3['Member_number'] = RFM_data['Member_number'].values
cluster_summary_3 = df_RFM_3.groupby('cluster').agg(
    Count = ('cluster', 'size'),
    Avg_Recency=('Recency', 'mean'),
    Avg_Frequency=('Frequency', 'mean'),
    Avg_Monetary=('Monetary', 'mean')
).reset_index()
cluster_summary_3['label'] = ['Cluster {}: {:.0f} customers\nAvg Recency: {:.0f} days\nAvg Frequency: {:.0f} orders\nAvg Monetary: ${:.0f}'.format(
    row['cluster'], row['Count'], row['Avg_Recency'], row['Avg_Frequency'], row['Avg_Monetary']) 
    for _, row in cluster_summary_3.iterrows()]

# For k=4
df_RFM_4 = df_RFM.copy()
df_RFM_4['cluster'] = kmeans_model.labels_  
df_RFM_4['Member_number'] = RFM_data['Member_number'].values

cluster_summary_4 = df_RFM_4.groupby('cluster').agg(
    Count = ('cluster', 'size'),
    Avg_Recency=('Recency', 'mean'),
    Avg_Frequency=('Frequency', 'mean'),
    Avg_Monetary=('Monetary', 'mean')
).reset_index()
cluster_summary_4['label'] = ['Cluster {}: {:.0f} customers\nAvg Recency: {:.0f} days\nAvg Frequency: {:.0f} orders\nAvg Monetary: ${:.0f}'.format(
    row['cluster'], row['Count'], row['Avg_Recency'], row['Avg_Frequency'], row['Avg_Monetary']) 
    for _, row in cluster_summary_4.iterrows()]

#-----------------HELPER-----------------#
# Function to create treemap
def plot_treemap(cluster_summary, title):
    cluster_summary['formatted_label'] = cluster_summary['label'].str.replace(" ", "\n")
    
    fig = go.Figure(go.Treemap(
        labels=cluster_summary['formatted_label'],
        parents=[""] * len(cluster_summary),
        values=cluster_summary['Count'],
        hovertemplate=(
            "Count: %{value}<br>"
            "Avg Recency: %{customdata[0]:.2f} days<br>"
            "Avg Frequency: %{customdata[1]:.2f} orders<br>"
            "Avg Monetary: %{customdata[2]:.2f} $<br>"
            "<extra></extra>"
        ),
        customdata=cluster_summary[['Avg_Recency', 'Avg_Frequency', 'Avg_Monetary']],
        marker=dict(colors=cluster_summary['Count'], colorscale='Viridis', showscale=False),
    ))

    fig.update_layout(
        title=title,
        margin=dict(t=40, l=20, r=20, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)


# Function to create 3D scatter plot
def plot_3d_scatter(df, kmeans_model, scaler, title):
    fig = px.scatter_3d(
        df, x='Recency', y='Frequency', z='Monetary',
        color='cluster',
        title=title,
        labels={'Recency': 'Recency', 'Frequency': 'Frequency', 'Monetary': 'Monetary'},
    )

    centroids = kmeans_model.cluster_centers_
    centers = scaler.inverse_transform(centroids)

    fig.add_scatter3d(
        x=centers[:, 0], y=centers[:, 1], z=centers[:, 2],
        mode='markers', marker=dict(size=12, color='red', symbol='x'),
        name='Cluster Centers'
    )
    fig.update_traces(marker=dict(size=2))
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis_title="Recency",
            yaxis_title="Frequency",
            zaxis_title="Monetary"
        )
    )

    st.plotly_chart(fig, use_container_width=True)


# Function to show recommendations based on predicted cluster and selected K value
def show_recommendations(predicted_cluster, k_value):
    if k_value == 4:
        if predicted_cluster == 0:
            st.write("##### Recommendations for Cluster 0: High Recency, Moderate Frequency, High Monetary")
            st.write("- Reward with VIP programs and exclusive offers.")
            st.write("- Recommend premium products based on past purchases.")
            st.write("- Implement personalized marketing strategies.")
        
        elif predicted_cluster == 1:
            st.write("##### Recommendations for Cluster 1: High Recency, Low Frequency, Low Monetary")
            st.write("- Send win-back campaigns with discounts or offers.")
            st.write("- Re-engage through personalized email reminders.")
            st.write("- Provide incentives for their next purchase.")

        elif predicted_cluster == 2:
            st.write("##### Recommendations for Cluster 2: Low Recency, High Frequency, High Monetary")
            st.write("- Encourage return with special deals or product bundles.")
            st.write("- Recommend new releases or loyalty rewards.")
            st.write("- Maintain engagement through exclusive promotions.")
        
        elif predicted_cluster == 3:
            st.write("##### Recommendations for Cluster 3: Moderate Recency, Low Frequency, Low Monetary")
            st.write("- Send low-cost incentives to encourage more purchases.")
            st.write("- Target with promotions to increase frequency of orders.")
            st.write("- Recommend budget-friendly products or starter packs.")

    elif k_value == 3:
        if predicted_cluster == 0:
            st.write("##### Recommendations for Cluster 0: Moderate Recency, Moderate Frequency, Low Monetary")
            st.write("- Encourage frequent purchases with promotions or discounts.")
            st.write("- Recommend value-oriented products.")
            st.write("- Offer incentives to drive repeat purchases.")
        
        elif predicted_cluster == 1:
            st.write("##### Recommendations for Cluster 1: High Recency, Low Frequency, Low Monetary")
            st.write("- Use win-back campaigns to bring them back.")
            st.write("- Provide incentives to encourage their next purchase.")
            st.write("- Send personalized offers based on past orders.")

        elif predicted_cluster == 2:
            st.write("##### Recommendations for Cluster 2: Low Recency, High Frequency, High Monetary")
            st.write("- Offer loyalty programs or exclusive offers for top spenders.")
            st.write("- Recommend premium or high-ticket items based on their past purchases.")
            st.write("- Create personalized marketing strategies to maintain engagement.")
        
    else:
        st.write("Invalid K value selected. Please choose either K=3 or K=4.")

def show_cluster_stats(cluster_id, k_value, transaction_df, product_df):
    st.write(f"## Cluster {cluster_id} Summary")

    if k_value == 3:
        cluster_members = df_RFM_3[df_RFM_3['cluster'] == cluster_id]['Member_number']
        cluster_transactions = transaction_df[transaction_df['Member_number'].isin(cluster_members)]
    else:
        cluster_members = df_RFM_4[df_RFM_4['cluster'] == cluster_id]['Member_number']
        cluster_transactions = transaction_df[transaction_df['Member_number'].isin(cluster_members)]

    # Merge with product data for product names
    merged_df = cluster_transactions.merge(product_df, on='productId')

    # Most purchased products
    top_products = merged_df['productName'].value_counts().head(5)
    st.write("##### 🔝 Top Products in this Cluster:")
    st.dataframe(top_products)

    # Find top product pairs
    pair_df = (
        merged_df.groupby(['Member_number', 'Date'])['productName']
        .apply(lambda x: list(set(x)))
        .reset_index()
    )


    all_pairs = Counter()
    for products in pair_df['productName']:
        if len(products) >= 2:
            pairs = combinations(sorted(products), 2)
            all_pairs.update(pairs)

    top_pairs = all_pairs.most_common(5)
    st.write("##### 🔗 Top Product Pairs in this Cluster:")
    for pair, count in top_pairs:
        st.write(f"- {pair[0]} & {pair[1]} — bought together {count} times")


# Main Function definitions
def overview():
    st.image("./img/banner.webp",use_container_width=True)
    st.title("Project Overview")
    st.write("""
        ### Overview of the Project
        This project is about building a recommendation system using clustering algorithms such as KMeans and Gaussian Mixture Models (GMM). The goal is to group customers based on their purchasing behaviors (Recency, Frequency, and Monetary).

        The project includes the following:  
        1. **Project Overview**: To imporve figures of a smaill shop by using customer clustering method to make strategies.    
        1. **Data Insight**: Analyzing the customer data for meaningful insights.
        2. **Modelling**: The process of implementing the clustering system.
        3. **New Prediction**: Making new predictions using the trained model.

        This tool will allow users to interact with the data, visualize clustering results, and make predictions for customer segmentation.
    """)
    st.markdown("---")
    st.write("### Algorithm")
    st.write("- **KMeans**: A popular clustering algorithm that partitions the data into k clusters, minimizing the variance within each cluster.")
    st.write("- **GMM (Gaussian Mixture Model)**: A probabilistic model that assumes all data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters.")
    st.image("./img/gmm_vs_kmeans_1.png",  use_container_width=True)
    
def datainsight():
    st.title('Data insight')

    # Product data frame
    with st.expander("##### Product data: "):
        col1, col2 = st.columns([2, 1])  
        col1.table(product_data.head(5))
        col2.write(f"""Include: 
                    \n ◉ {product_data.shape[0]} rows 
                    \n ◉ {product_data.shape[1]} columns """)

    # Transaction data frame
    with st.expander("##### Transaction data: "):
        col1, col2 = st.columns([2, 1])  
        col1.table(transaction_data.head(5))
        col2.write(f"""Include: 
                    \n ◉ {transaction_data.shape[0]} rows 
                    \n ◉ {transaction_data.shape[1]} columns """)
    
    # Final data frame
    with st.expander("##### Final data: "):
        col1, col2 = st.columns([2, 1])  
        col1.table(RFM_data.head(5))
        col2.write(f"""Include: 
                    \n ◉ {RFM_data.shape[0]} rows 
                    \n ◉ {RFM_data.shape[1]} columns """)
   

    # Data visualization 
    st.subheader("📊 Overview Stats")

    # Prepare data
    total_customers = RFM_data['Member_number'].nunique()
    total_products = product_data['productId'].nunique()
    total_categories = product_data['Category'].nunique()
    total_transactions = transaction_data.shape[0]
    total_revenue = (transaction_data.merge(product_data, on='productId')['price'] * transaction_data['items']).sum()

    # Row 1: 3 metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 👥 Total Customers")
        st.success(f"**{total_customers:,}**")

    with col2:
        st.markdown("#### 📦 Total Products")
        st.success(f"**{total_products:,}**")

    with col3:
        st.markdown("#### 🗂️ Total Categories")
        st.success(f"**{total_categories:,}**")

    st.markdown("---")  

    col4, col5 = st.columns(2)
    with col4:
        st.markdown("#### 🧾 Total Transactions")
        st.info(f"**{total_transactions:,}**")

    with col5:
        st.markdown("#### 💰 Total Revenue")
        st.info(f"**${total_revenue:,.0f}**")


    st.subheader("🏆 Top Selling Products by Revenue")
    merged = transaction_data.merge(product_data, on='productId')
    product_sales = merged.groupby('productName')['items'].sum().sort_values(ascending=False).head(10)
    st.bar_chart(product_sales)

    revenue_by_cat = merged.copy()
    revenue_by_cat['revenue'] = revenue_by_cat['items'] * revenue_by_cat['price']
    cat_revenue = revenue_by_cat.groupby('Category')['revenue'].sum().sort_values(ascending=False)
    st.subheader("💼 Revenue by Category")
    st.bar_chart(cat_revenue)

    merged['date'] = pd.to_datetime(merged['Date'],dayfirst=True)
    sales_trend = merged.copy()
    sales_trend['revenue'] = sales_trend['items'] * sales_trend['price']
    daily_sales = sales_trend.groupby('Date')['revenue'].sum()

    st.subheader("📅 Sales Trend Over Time")
    st.line_chart(daily_sales)

    st.subheader("📦 Customer Purchase Frequency")

    customer_orders = transaction_data['Member_number'].value_counts()

    fig, ax = plt.subplots()
    ax.hist(customer_orders, bins=20, color='skyblue', edgecolor='black')
    ax.set_title("Customer Purchase Frequency")
    ax.set_xlabel("Number of Purchases")
    ax.set_ylabel("Number of Customers")

    st.pyplot(fig)
    st.subheader("💰 Customer Lifetime Value (LTV)")

    # Merge price with transactions
    merged = transaction_data.merge(product_data[['productId', 'price']], on='productId')
    merged['revenue'] = merged['items'] * merged['price']

    # revenue per customer
    ltv_df = merged.groupby('Member_number')['revenue'].sum().reset_index().sort_values(by='revenue', ascending=False)
    ltv_df.columns = ['Customer', 'Expenditure']
    st.dataframe(ltv_df.head(10))


    # Top 10 Products
    top_products = raw_data.groupby('productName')['items'].sum().sort_values(ascending=True).head(10)

    with col4:
        fig, ax = plt.subplots(figsize=(6, 4))
        top_products.plot(kind='barh', ax=ax, color='skyblue', title="Top 10 Products", edgecolor='black')
        ax.set_xlabel('Items Sold')
        ax.set_ylabel('Product Name')
        ax.set_title('Top 10 Products', fontsize=24)
        plt.tight_layout()
        st.pyplot(fig)  

    # Top 10 Categories
    top_categories = (raw_data.groupby('Category')['items'].sum().sort_values(ascending=False).head(10).reset_index()) 
    with col5:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(top_categories['Category'], top_categories['items'], edgecolor='black', color='lightgreen')
        ax.set_xlabel('Items Sold')
        ax.set_ylabel('Category')
        ax.set_title('Top 10 Category ', fontsize=24)
        ax.invert_yaxis()
        plt.tight_layout()

        # Display in Streamlit
        st.pyplot(fig)
    
    st.subheader("💰 Revenue Comparison: Expensive vs Cheap Products")

    st.dataframe(compare_df)

    # Plot revenue
    fig = px.bar(compare_df, x='productName', y='total_revenue',
                color='type', barmode='group',
                title="Total Revenue: Most Expensive vs Cheapest Products",
                labels={'total_revenue': 'Revenue ($)', 'productName': 'Product'})
    st.plotly_chart(fig)
    fig2 = px.bar(compare_df, x='productName', y='quantity_sold',
              color='type', barmode='group',
              title="Quantity Sold Comparison",
              labels={'quantity_sold': 'Items Sold', 'productName': 'Product'})
    st.plotly_chart(fig2)

    st.subheader(" 📈 RFM Relationship")
    st.write("Monetary and Frequency have a close relationship; when frequency increases, monetary also increases.")

    corr_matrix = RFM_data[['Frequency', 'Monetary', 'Recency']].corr()

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title("Correlation between Frequency, Monetary, and Recency")
    st.pyplot(plt)


def models():
    # Result from KMeans and GMM
    st.write("#### Modelling with KMeans and GMM")
    st.subheader("Models")
    st.markdown('<h5 style="color: #FF6347;">With K = 3</h5>', unsafe_allow_html=True)
    st.image("./img/score_3.png", use_container_width=True)
    st.markdown('<h5 style="color: #FF6347;">With K = 4</h5>', unsafe_allow_html=True)
    st.image("./img/K.png", use_container_width=True)
    st.write("**Note**: In the Elbow Method, K = 3 or K = 4 is a visible elbow, suggesting a good trade-off between model complexity and performance.While the Silhouette Score slightly drops at K = 3 or K = 4, it’s still reasonably high, meaning the clusters remain fairly distinct.")

    
    st.subheader("Customer Segments Visualization")
    st.markdown('<h5 style="color: #FF6347;">With K = 3</h5>', unsafe_allow_html=True)
    col6, col7 = st.columns(2)

    with col6:
        plot_treemap(cluster_summary_3, "Customer Segments - Treemap")

    with col7:
        plot_3d_scatter(df_RFM_3, kmeans_model_3, scaler, "Customer Segmentation with KMeans (3D)")

    st.markdown('<h5 style="color: #FF6347;">With K = 4</h5>', unsafe_allow_html=True)
    col8, col9 = st.columns(2)

    with col8:
        plot_treemap(cluster_summary_4, "Customer Segments - Treemap")

    with col9:
        plot_3d_scatter(df_RFM_4, kmeans_model, scaler, "Customer Segmentation with KMeans (3D)")

def Prediction():
    st.title("Prediction")
    st.write("##### Predict from Slider (Single Input using KMeans)")

    recency = st.slider("Recency", 1, round(max(RFM_data['Recency'])), 1)
    frequency = st.slider("Frequency", 1, round(max(RFM_data['Frequency'])), 1)
    monetary = st.slider("Monetary", 1, round(max(RFM_data['Monetary'])), 1)
    k_slider = st.radio("Choose number of clusters (k):", [3, 4], key="k_slider")

    st.write(f"Recency: {recency}, Frequency: {frequency}, Monetary: {monetary} with k = {k_slider}")
    
    if st.button("Predict Cluster from Slider Inputs"):
        input_data = np.array([[recency, frequency, monetary]])
        scaled_input = scaler.transform(input_data)
        
        if k_slider == 4:
            kmeans_pred = kmeans_model.predict(scaled_input)
            cluster_info =  cluster_summary_4[cluster_summary_4['cluster'] == kmeans_pred[0]].iloc[0]
        else:
            kmeans_pred = kmeans_model_3.predict(scaled_input)
            cluster_info = cluster_summary_3[cluster_summary_3['cluster'] == kmeans_pred[0]].iloc[0]

        st.write(f"Predicted Cluster: {kmeans_pred[0]}")
        st.write(f"##### Cluster {kmeans_pred[0]} Characteristics:")
        st.write(f"- **Average Recency**: {round(cluster_info['Avg_Recency'])} days")
        st.write(f"- **Average Frequency**: {round(cluster_info['Avg_Frequency'])} orders")
        st.write(f"- **Average Monetary**: ${round(cluster_info['Avg_Monetary'])}")
       
        show_cluster_stats(kmeans_pred[0],k_slider,transaction_data,product_data)
        show_recommendations(kmeans_pred[0],k_slider)
    
    st.write("---")
    st.write("##### Upload file (Batch Prediction using GMM)")

    k_file = st.radio("Choose number of clusters for file (k):", [3, 4], key="k_file")


    # Predefined file 
    predefined_file_path = "./data/df_RFM.csv"

    file_option = st.radio("Choose file source", ("Use predefined file","Upload CSV file"))

    if file_option == "Upload CSV file":
        upload_file = st.file_uploader("Upload CSV file", type=['csv'])

        if upload_file is not None:
            if upload_file.name.endswith('csv'):
                rfm_input = pd.read_csv(upload_file)

                required_columns = {'Member_number', 'Recency', 'Frequency', 'Monetary'}
                if required_columns.issubset(rfm_input.columns):
                    scaled_input = scaler.transform(rfm_input[['Recency', 'Frequency', 'Monetary']])

                    if k_file == 4:
                        rfm_input['cluster'] = gmm_model.predict(scaled_input)
                    else:
                        rfm_input['cluster'] = gmm_model_3.predict(scaled_input)

                    st.write("Clustered Members from Uploaded File:")
                    st.dataframe(rfm_input.head(5))
                else:
                    st.error("The file must have these columns: 'Member_number', 'Recency', 'Frequency', 'Monetary'")
            else:
                st.error("Unsupported file type. Please upload a CSV file.")

    elif file_option == "Use predefined file":
        rfm_input = pd.read_csv(predefined_file_path)

        required_columns = {'Member_number', 'Recency', 'Frequency', 'Monetary'}
        if required_columns.issubset(rfm_input.columns):
            scaled_input = scaler.transform(rfm_input[['Recency', 'Frequency', 'Monetary']])

            if k_file == 4:
                rfm_input['cluster'] = gmm_model.predict(scaled_input)
            else:
                rfm_input['cluster'] = gmm_model_3.predict(scaled_input)

            st.write("Clustered Members from Predefined File:")
            st.dataframe(rfm_input.head(5))
        else:
            st.error("The file must have these columns: 'Member_number', 'Recency', 'Frequency', 'Monetary'")

# Create the pages and menu
pages = {
    "Menu" : [
        st.Page(overview, title="Project Overview", icon="🔥"),
        st.Page(datainsight, title="Data Insight", icon="📊"),
        st.Page(models, title="Modelling", icon="💡"),
        st.Page(Prediction, title="Prediction", icon="🎯"),
    ]
}

# Navigation and page selection
pg = st.navigation(pages)
pg.run()

# Sidebar information
st.sidebar.markdown("**👨‍🎓Group:**")
st.sidebar.markdown("Huynh Triet & Hoang Huyen")  
st.sidebar.markdown("**👩🏼‍🏫Teacher:**")
st.sidebar.markdown("Dr. Khuat Thuy Phuong")  
st.sidebar.markdown("**🗓️Timeline:**")
st.sidebar.markdown("20/4/2025")
