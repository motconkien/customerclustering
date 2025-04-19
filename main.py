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

st.set_page_config(layout="wide") #expand layout
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
compare_df = compare_df.sort_values(by="price", ascending=False).reset_index()

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
df_RFM_3['Cluster'] = kmeans_model_3.labels_
df_RFM_3['Member_number'] = RFM_data['Member_number'].values
cluster_summary_3 = df_RFM_3.groupby('Cluster').agg(
    Count = ('Cluster', 'size'),
    Avg_Recency=('Recency', 'mean'),
    Avg_Frequency=('Frequency', 'mean'),
    Avg_Monetary=('Monetary', 'mean')
).reset_index()
cluster_summary_3['label'] = ['Cluster {}: {:.0f} customers\nAvg Recency: {:.0f} days\nAvg Frequency: {:.0f} orders\nAvg Monetary: ${:.0f}'.format(
    row['Cluster'], row['Count'], row['Avg_Recency'], row['Avg_Frequency'], row['Avg_Monetary']) 
    for _, row in cluster_summary_3.iterrows()]

# For k=4 - kmeans 
df_RFM_4 = df_RFM.copy()
df_RFM_4['Cluster'] = kmeans_model.labels_  
df_RFM_4['Member_number'] = RFM_data['Member_number'].values

cluster_summary_4 = df_RFM_4.groupby('Cluster').agg(
    Count = ('Cluster', 'size'),
    Avg_Recency=('Recency', 'mean'),
    Avg_Frequency=('Frequency', 'mean'),
    Avg_Monetary=('Monetary', 'mean')
).reset_index()
cluster_summary_4['label'] = ['Cluster {}: {:.0f} customers\nAvg Recency: {:.0f} days\nAvg Frequency: {:.0f} orders\nAvg Monetary: ${:.0f}'.format(
    row['Cluster'], row['Count'], row['Avg_Recency'], row['Avg_Frequency'], row['Avg_Monetary']) 
    for _, row in cluster_summary_4.iterrows()]

# For k=4 - gmm 
df_RFM_4_gmm = df_RFM.copy()
df_RFM_4_gmm['Cluster'] = kmeans_model.labels_  
df_RFM_4_gmm['Member_number'] = RFM_data['Member_number'].values

cluster_summary_4_gmm = df_RFM_4_gmm.groupby('Cluster').agg(
    Count = ('Cluster', 'size'),
    Avg_Recency=('Recency', 'mean'),
    Avg_Frequency=('Frequency', 'mean'),
    Avg_Monetary=('Monetary', 'mean')
).reset_index()
cluster_summary_4_gmm['label'] = ['Cluster {}: {:.0f} customers\nAvg Recency: {:.0f} days\nAvg Frequency: {:.0f} orders\nAvg Monetary: ${:.0f}'.format(
    row['Cluster'], row['Count'], row['Avg_Recency'], row['Avg_Frequency'], row['Avg_Monetary']) 
    for _, row in cluster_summary_4_gmm.iterrows()]

#-----------------------------------------------------------------HELPER---------------------------------------------------------------------------------#
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
        textfont=dict(size=24)
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
        color='Cluster',
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
    fig.update_traces(marker=dict(size=1))
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


# Function to show recommendations based on predicted Cluster and selected K value
def show_recommendations(predicted_cluster):
    if predicted_cluster == 0:
        st.write("##### Recommendations for Cluster 0  - Normal: High Recency, Moderate Frequency, High Monetary")
        st.write("- Reward with VIP programs and exclusive offers.")
        st.write("- Recommend premium products based on past purchases.")
        st.write("- Implement personalized marketing strategies.")
        
    elif predicted_cluster == 1:
        st.write("##### Recommendations for Cluster 1 - Risk: High Recency, Low Frequency, Low Monetary")
        st.write("- Send win-back campaigns with discounts or offers.")
        st.write("- Re-engage through personalized email reminders.")
        st.write("- Provide incentives for their next purchase.")

    elif predicted_cluster == 2:
        st.write("##### Recommendations for Cluster 2 - Vip: Low Recency, High Frequency, High Monetary")
        st.write("- Encourage return with special deals or product bundles.")
        st.write("- Recommend new releases or loyalty rewards.")
        st.write("- Maintain engagement through exclusive promotions.")
        
    elif predicted_cluster == 3:
        st.write("##### Recommendations for Cluster 3 - New: Moderate Recency, Low Frequency, Low Monetary")
        st.write("- Send low-cost incentives to encourage more purchases.")
        st.write("- Target with promotions to increase frequency of orders.")
        st.write("- Recommend budget-friendly products or starter packs.")
    else:
        return

def get_cluster_name(Cluster):
    if Cluster == 1:
        return "Risk"
    elif Cluster == 2:
        return "VIP"
    elif Cluster == 3:
        return "New"
    else:
        return "Normal"
    
def show_cluster_stats(cluster_id, model, transaction_df, product_df):
    
    st.write(f"###### Cluster {cluster_id} ({get_cluster_name(cluster_id)}) Summary")

    if model == "Kmeans":
        cluster_members = df_RFM_4[df_RFM_4['Cluster'] == cluster_id]['Member_number']
        cluster_transactions = transaction_df[transaction_df['Member_number'].isin(cluster_members)]
    else:
        cluster_members = df_RFM_4_gmm[df_RFM_4_gmm['Cluster'] == cluster_id]['Member_number']
        cluster_transactions = transaction_df[transaction_df['Member_number'].isin(cluster_members)]

    merged_df = cluster_transactions.merge(product_df, on='productId')

    # Most purchased products
    top_products = (
        merged_df.groupby(['productName', 'price'])
        .size()
        .reset_index(name='purchase_count')
        .sort_values(by='purchase_count', ascending=False)
        .head(5)
    )
    st.write("###### 🔝 Top Products in this Cluster:")
    #change positions, should put price afer? 
    st.dataframe(top_products[['productName', 'purchase_count', 'price']])

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
    st.write("###### 🔗 Top Product Pairs in this Cluster:")
    for pair, count in top_pairs:
        st.write(f"- {pair[0]} & {pair[1]} — bought together {count} times")



#----------------------------------------------------------------------------------------------------------------------------------------------------#
# Main Function definitions
def overview():
    st.image("./img/banner.webp",use_container_width=True)
    st.title("Project Overview")
    
    # Background
    with st.container():
        st.subheader("Customer segmentation")
        st.markdown("""
        Customer segmentation is a fundamental approach in data science that enables businesses to:
        - Understand different customer profiles based on purchasing behavior,
        - Personalize marketing strategies and product offerings,
        - Enhance customer satisfaction and retention.

        """)

    # Business Objective
    with st.container():
        st.subheader("Business Objective / Problem")
        st.markdown("""
        For Retail Store X, applying customer segmentation helps make informed decisions in product stocking, marketing, and customer care.

        The store owner of Retail Store X aims to:
        - Understand customer behavior from transactional data,
        - Recommend the right products to the right customer groups,
        - Improve customer service quality and increase customer satisfaction.
        """)

    # Proposed Solution
    with st.container():
        st.subheader("💡 Proposed Solution")
        st.markdown("""
        To address the business goals, we apply **Data Science** by:
        - Using RFM (Recency, Frequency, Monetary) analysis to quantify customer behavior,
        - Applying clustering algorithms to segment customers into meaningful groups.
        """)

    # Models Used
    with st.container():
        st.subheader("⚙️ Models Applied")
        st.markdown("""
        - **IQR Rule-Based Segmentation**  
        - **KMeans** (Scikit-learn and PySpark)  
        - **Gaussian Mixture Model (GMM)**  
        - **Hierarchical Clustering**
        """)

    # Key Result
    with st.container():
        st.subheader("✅ Key Result")
        st.markdown("""
        Among the models tested, Scikit-learn KMeans produced a balanced segmentation with 4 distinct clusters, achieving a Silhouette Score of 0.38, which indicates a moderate but effective separation between customer groups.
        """)

    with st.container():
        st.subheader("🖥️ Streamlit Application Output")
        st.markdown("""
        This project is deployed as a **Streamlit web application**, enabling interactive customer segmentation. 

        **Main Features of the App:**
        - **Overview**: Presenting the project goals, methodology, and results.
        - **Data Insight**: Visual exploration of RFM metrics and customer behavior.
        - **Modeling**: Apply clustering algorithms (IQR, KMeans, GMM, Hierarchical) to segment customers.
        - **New Prediction**: Users can input new RFM values and receive real-time predictions of customer cluster membership.

        The tool supports both data analysis and decision-making by allowing users to:
        - Visualize cluster structures,
        - Upload new data for segmentation,
        - Improve business strategy based on insights.

        This interactive environment helps bridge data science with business strategy, making customer segmentation accessible for small business owners.
        """)
    
    
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
    st.markdown("""
    <style>
    .big-font {
        font-size:40px !important;
        background: #41644A;
        color:#FFFFFF;
        border-radius: 10px;
        padding-left: 10px;
        
    }

    </style>
    """, unsafe_allow_html=True)
    with col1:
        st.markdown("#### 👥 Total Customers")
        # st.success(f"**{total_customers:,}**")
        st.markdown(f'<p class="big-font">{total_customers}</p>', unsafe_allow_html=True)

    with col2:
        st.markdown("#### 📦 Total Products")
        st.markdown(f'<p class="big-font">{total_products:,}</p>', unsafe_allow_html=True)

    with col3:
        st.markdown("#### 🗂️ Total Categories")
        st.markdown(f'<p class="big-font">{total_categories:,}</p>', unsafe_allow_html=True)



    col4, col5 = st.columns(2)
    with col4:
        st.markdown("#### 🧾 Total Transactions")
        st.markdown(f'<p class="big-font">{total_transactions:,}</p>', unsafe_allow_html=True)

    with col5:
        st.markdown("#### 💰 Total Revenue")
        st.markdown(f'<p class="big-font">{total_revenue:,.0f}</p>', unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("💎 Top Selling Products by Quantity")
    col6, col7 = st.columns(2)
    # Top 10 Products
    top_products = raw_data.groupby('productName')['items'].sum().sort_values(ascending=True).head(10)

    with col6:
        fig, ax = plt.subplots(figsize=(6, 4))
        top_products.plot(kind='barh', ax=ax, color='skyblue', title="Top 10 Products", edgecolor='black')
        ax.set_xlabel('Items Sold')
        ax.set_ylabel('Product Name')
        ax.set_title('Top 10 Products', fontsize=24)
        plt.tight_layout()
        st.pyplot(fig)  

    # Top 10 Categories
    top_categories = (raw_data.groupby('Category')['items'].sum().sort_values(ascending=False).head(10).reset_index()) 
    with col7:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(top_categories['Category'], top_categories['items'], edgecolor='black', color='lightgreen')
        ax.set_xlabel('Items Sold')
        ax.set_ylabel('Category')
        ax.set_title('Top 10 Categories ', fontsize=24)
        ax.invert_yaxis()
        plt.tight_layout()

        st.pyplot(fig)
    st.markdown("---")  
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


    merged = transaction_data.merge(product_data[['productId', 'price']], on='productId')
    merged['revenue'] = merged['items'] * merged['price']

    # revenue per customer
    ltv_df = merged.groupby('Member_number')['revenue'].sum().reset_index().sort_values(by='revenue', ascending=False)
    ltv_df.columns = ['Customer', 'Expenditure']
    st.dataframe(ltv_df.head(10))

    
    
    #-------------------The most expensive and cheapest products------------------#
    st.subheader("💰 Revenue Comparison: Expensive vs Cheap Products")

    st.dataframe(compare_df)
    #for revenue
    col8,col9 = st.columns(2)
    with col8:
        figg = go.Figure()
        bar_colors = compare_df['type'].map({
        'Most Expensive': 'darkblue',
        'Cheapest': 'green'
        })
        # Bar chart for revenue sold
        figg.add_trace(go.Bar(
            x=compare_df['productName'],
            y=compare_df['total_revenue'],
            name='Total Revenue',
            yaxis='y1',
            showlegend=False,
            marker=dict(color=bar_colors)
        ))

        # Line chart for unit price
        figg.add_trace(go.Scatter(
            x=compare_df['productName'],
            y=compare_df['price'],
            name='Unit Price',
            mode='lines+markers',
            line=dict(color='red', width=3),
            marker=dict(size=8),
            yaxis='y2'
            
        ))

        # Layout with dual y-axes
        figg.update_layout(
            title="Total Revenue vs Unit Price (Most Expensive & Cheapest Products)",
            xaxis=dict(title="Product"),
            yaxis=dict(
                title="Revenue ($)",
                side="left"
            ),
            yaxis2=dict(
                title="Unit Price ($)",
                overlaying="y",
                side="right",
                color='red'
            ),
            # legend_title="Legend",
            template='simple_white',
            barmode='group',
            xaxis_tickangle=45,
            # showlegend = True
        )
        figg.add_trace(go.Bar(
            x=[None], y=[None],
            name='Cheapest',
            marker=dict(color='green'),
            showlegend=True
        ))
        figg.add_trace(go.Bar(
            x=[None], y=[None],
            name='Most Expensive',
            marker=dict(color='darkblue'),
            showlegend=True
        ))
        st.plotly_chart(figg, use_container_width=True)
        #for quantity
        fig = go.Figure()

    with col9:
    # Bar chart for quantity sold
        fig.add_trace(go.Bar(
            x=compare_df['productName'],
            y=compare_df['quantity_sold'],
            name='Total Quantity Sold',
            yaxis='y1',
            showlegend=False,
            marker=dict(color=bar_colors)
        ))

        # Line chart for unit price
        fig.add_trace(go.Scatter(
            x=compare_df['productName'],
            y=compare_df['price'],
            name='Unit Price',
            mode='lines+markers',
            line=dict(color='red', width=3),
            marker=dict(size=8),
            yaxis='y2'
        ))

        # Layout with dual y-axes
        fig.update_layout(
            title="Quantity Sold vs Unit Price (Most Expensive & Cheapest Products)",
            xaxis=dict(title="Product"),
            yaxis=dict(
                title="Quantity Sold",
                side="left"
            ),
            yaxis2=dict(
                title="Unit Price ($)",
                overlaying="y",
                side="right",
                
            ),
            legend_title="Legend",
            template='simple_white',
            barmode='group',
            xaxis_tickangle=45,
           
        )
        fig.add_trace(go.Bar(
            x=[None], y=[None],
            name='Cheapest',
            marker=dict(color='green'),
            showlegend=True
        ))
        fig.add_trace(go.Bar(
            x=[None], y=[None],
            name='Most Expensive',
            marker=dict(color='darkblue'),
            showlegend=True
        ))
        st.plotly_chart(fig, use_container_width=True)
    st.write("Though bottled water has the much quantity sole but because the cheap product it is, the revenue is low. " \
    "In addition, napskin is the expensive product and the quantity sold is moderate, the revenue is more significant than others")

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
    st.write("### Modelling with KMeans and GMM")
    st.markdown("---")
    st.write("#### 1. Algorithm")
    st.write("- **KMeans**: A popular clustering algorithm that partitions the data into k clusters, minimizing the variance within each Cluster. ")
    st.write("- **GMM (Gaussian Mixture Model)**: A probabilistic model that assumes all data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters.")
    st.image("./img/gmm_vs_kmeans_1.png",  use_container_width=True, caption="GMM and Kmeans in Clustering")
    st.markdown("---")
    st.write("#### 2. Models")
    st.markdown('<h5 style="color: #FF6347;">Kmeans with K = 4</h5>', unsafe_allow_html=True)
    st.image("./img/kmeans_mod.png", use_container_width=True, caption="Kmeans Scores Graph: Elbow Method, Silhouette Score, Davies-Bouldin Index")
    st.markdown('<h5 style="color: #FF6347;">GMM with K = 5</h5>', unsafe_allow_html=True)
    st.image("./img/gmm_mod.png", use_container_width=True, caption="GMM Scores Graph: Silhouette Score, BIC Score, AIC Score")
    st.markdown("---")
    st.write("#### 3. Model Evaluation Metrics")
    metrics = {
    'Model': ['Kmeans', 'GMM'],
    'K (Optimal)': [4,5],
    'Silhouette Score': [0.33, 0.19],
    'Model Training Time': [0.066, 2.828],
    'Prediction Time': [0.063, 0.03]
}
    df_metrics = pd.DataFrame(metrics)

    st.dataframe(df_metrics, use_container_width=True)
    
    st.markdown("""
    **Conclusion**  
    - Because Silhouette Score of Kmeans is higher than that of GMM
    - The training time and prediction time of KMeans are shorter than those of GMM \n
    => :blue[Kmeans is the best choice in this case]       
    """)

    st.write("---")
    st.write("#### 4. Customer Segments Visualization")
   # KMeans results section
    st.markdown('<h5 style="color: #FF6347;">KMeans with K = 4</h5>', unsafe_allow_html=True)
    col6, col7 = st.columns(2)

    with col6:
        plot_treemap(cluster_summary_4, "Customer Segments - Treemap")

    with col7:
        plot_3d_scatter(df_RFM_4, kmeans_model, scaler, "Customer Segmentation with KMeans (3D)")



#prediction page: need to change k =3,4 to choose models
# summary first / file / ID/ -> summary per Cluster
def Prediction():
    st.title("Prediction")


    #predict via files
    # Predefined file 
    st.write("##### Upload file")
    file_option = st.radio("Choose file source", ("Use predefined file","Upload CSV file"))

    # model_choose = st.radio("Choose model for Cluster prediction:", ["Kmeans", "GMM"], key="model")
    predefined_file_path = "./data/df_RFM.csv"

    if st.button("Show cluster"):
        #summary
        
        
        st.write("###### Clusters Summary")
        cluster_summary_4['Type'] = cluster_summary_4['Cluster'].apply(get_cluster_name)
        for col in ['Avg_Recency', 'Avg_Frequency', 'Avg_Monetary']:
            cluster_summary_4[col] = cluster_summary_4[col].round(2)
        df_transposed = cluster_summary_4[['Cluster', 'Type', 'Count', 'Avg_Recency', 'Avg_Frequency', 'Avg_Monetary']].T
        df_transposed.columns = df_transposed.iloc[0]  
        df_transposed = df_transposed[1:]  
        df_transposed = df_transposed.reset_index().rename(columns={'index': 'Cluster'})
        st.dataframe(df_transposed)

        if file_option == "Upload CSV file":
            upload_file = st.file_uploader("Upload CSV file", type=['csv'])

            
            if upload_file is not None:
                if upload_file.name.endswith('csv'):
                    rfm_input = pd.read_csv(upload_file)

                    required_columns = {'Member_number', 'Recency', 'Frequency', 'Monetary'}
                    if required_columns.issubset(rfm_input.columns):
                        scaled_input = scaler.transform(rfm_input[['Recency', 'Frequency', 'Monetary']])

                        # if model_choose == "Kmeans":
                        rfm_input['Cluster'] = kmeans_model.predict(scaled_input)
                        rfm_input['Type'] = rfm_input['Cluster'].apply(get_cluster_name)
                        # else:
                        #     rfm_input['Cluster'] = gmm_model.predict(scaled_input)
                        #     rfm_input['Type'] = rfm_input['Cluster'].apply(get_cluster_name)

                        st.write("Clustered Members from Uploaded File:")
                        st.dataframe(rfm_input)
                    else:
                        st.error("The file must have these columns: 'Member_number', 'Recency', 'Frequency', 'Monetary'")
                else:
                    st.error("Unsupported file type. Please upload a CSV file.")

        elif file_option == "Use predefined file":
            rfm_input = pd.read_csv(predefined_file_path)

            required_columns = {'Member_number', 'Recency', 'Frequency', 'Monetary'}
            if required_columns.issubset(rfm_input.columns):
                scaled_input = scaler.transform(rfm_input[['Recency', 'Frequency', 'Monetary']])

                
                rfm_input['Cluster'] = kmeans_model.predict(scaled_input)
                rfm_input['Type'] = rfm_input['Cluster'].apply(get_cluster_name)
                # else:
                #     rfm_input['Cluster'] = gmm_model.predict(scaled_input)
                #     rfm_input['Type'] = rfm_input['Cluster'].apply(get_cluster_name)

                st.write("Clustered Members from Predefined File:")
                st.dataframe(rfm_input)
            else:
                st.error("The file must have these columns: 'Member_number', 'Recency', 'Frequency', 'Monetary'")
        
        
    # st.write("---")
    #input id
    st.write("---")
    st.write("##### Customer Behavior Based On")
    option = st.radio("Chose one of options below: ", options=['Customer ID', 'RFM Slider'])
    if option == 'Customer ID':
        st.write("###### Customer Behavior Based on Customer ID")

        id = st.number_input("Enter customer ID", step=1, format="%d")
        # model_id = st.radio("Choose model for Cluster prediction:", ["Kmeans", "GMM"], key="mod")
        st.write(f"ID Suggestion: {[int(i) for i in np.random.choice(RFM_data['Member_number'].values, 5, replace=False)]}")

        if id in RFM_data['Member_number'].values:
            
            customer_df = df_RFM_4[df_RFM_4['Member_number'] == id]
            

            customer_df['Type'] = customer_df['Cluster'].apply(get_cluster_name)
            customer_df = customer_df[['Member_number', 'Cluster', 'Type', 'Recency', 'Frequency', 'Monetary']]
            st.dataframe(customer_df)
            #show recommend
            st.write(f"##### Customer ID {id} :blue[{customer_df.iloc[0]['Cluster']} - {get_cluster_name(customer_df.iloc[0]['Cluster'])} customer]")
            show_cluster_stats(customer_df.iloc[0]['Cluster'],'Kmeans',transaction_data,product_data)
            show_recommendations(customer_df.iloc[0]['Cluster'])

        else:
            st.warning("Customer ID not found! Please enter a valid ID from the dataset.")
        
    st.write("---")
    if option == 'RFM':
    #Input stats
        st.write("###### Customer Behavior Based on Recency, Frequency, and Monetary (RFM) Data")

        recency = st.slider("Recency", 1, round(max(RFM_data['Recency'])), 1)
        frequency = st.slider("Frequency", 1, round(max(RFM_data['Frequency'])), 1)
        monetary = st.slider("Monetary", 1, round(max(RFM_data['Monetary'])), 1)
        # model_slider = st.radio("Choose model for Cluster prediction:", ["Kmeans", "GMM"], key="slider")

        st.write(f"Recency: {recency}, Frequency: {frequency}, Monetary: {monetary} ")
        
        if st.button("Predict Cluster from Slider Inputs"):
            input_data = np.array([[recency, frequency, monetary]])
            scaled_input = scaler.transform(input_data)
            
        
            kmeans_pred = kmeans_model.predict(scaled_input)
            cluster_info =  cluster_summary_4[cluster_summary_4['Cluster'] == kmeans_pred[0]].iloc[0]
        

            st.write(f"##### Predicted Cluster: :blue[{kmeans_pred[0]} - {get_cluster_name(kmeans_pred[0])} customer]")
            st.write(f"###### Cluster {kmeans_pred[0]} -  Characteristics:")
            st.write(f"- **Average Recency**: {round(cluster_info['Avg_Recency'])} days")
            st.write(f"- **Average Frequency**: {round(cluster_info['Avg_Frequency'])} orders")
            st.write(f"- **Average Monetary**: ${round(cluster_info['Avg_Monetary'])}")
        
            show_cluster_stats(kmeans_pred[0],"Kmeans",transaction_data,product_data)
            show_recommendations(kmeans_pred[0])

        st.write("---")
    st.markdown("### 🆕 Customer Segment Analysis")
    if st.button("Customer Segment Analysis"):
       
        # 1. Assign Segment Labels
        cluster_labels = {
            0: 'Normal',
            1: 'Risk',
            2: 'VIP',
            3: 'New'
        }
        df_RFM_4['Segment'] = df_RFM_4['Cluster'].map(cluster_labels)

        # 2. Merge Transactions with Segment & Product Info
        transaction_data_merged = (
            transaction_data
            .merge(df_RFM_4[['Member_number', 'Segment']], on='Member_number', how='inner')
            .merge(product_data, on='productId', how='left')
        )

        # 3. Aggregate Total Items per Product per Segment
        product_by_segment = (
            transaction_data_merged.groupby(['Segment', 'productName'])['items']
            .sum().reset_index()
        )

        # 4. Remove Duplicated Product Names Across Segments, Keep Highest
        unique_products = (
            product_by_segment.sort_values(['productName', 'items'], ascending=[True, False])
            .drop_duplicates(subset='productName', keep='first')
        )

        # 5. Helper Function to Get Top/Bottom Products
        def get_top_bottom_products(df, n=5, top=True, label="Top"):
            df_sorted = df.sort_values(['Segment', 'items'], ascending=[True, not top])
            top_n = df_sorted.groupby('Segment').head(n).copy()
            top_n['Rank'] = top_n.groupby('Segment').cumcount() + 1
            pivot_df = top_n.pivot(index='Rank', columns='Segment', values='productName')
            pivot_df.index.name = f'{label} Rank'
            pivot_df.columns.name = None
            return pivot_df, top_n

        # 6. Top & Bottom Products
        top5_table, top5_popular = get_top_bottom_products(unique_products, n=5, top=True, label="Top")
        bottom5_table, bottom5_least = get_top_bottom_products(unique_products, n=5, top=False, label="Least")

        # st.markdown('#### ⭐ Top 5 Most Popular Products per Segment')
        # st.dataframe(top5_table)

        # st.markdown('#### 🔻 Bottom 5 Least Popular Products per Segment')
        # st.dataframe(bottom5_table)

        # 7. Boost Suggestions
        st.markdown('#### 🚀 Top 10 Boost Suggestions for Revenue Growth')
        st.markdown("> High-performing products with strong sales and/or high unit price. Prioritize for promotion to drive revenue.")

        boost_candidates = top5_popular[top5_popular['Segment'].isin(['VIP', 'Normal'])].copy()
        boost_candidates = boost_candidates.merge(product_data[['productName', 'price']], on='productName', how='left')
        boost_candidates['revenue'] = boost_candidates['items'] * boost_candidates['price']
        boost_candidates = boost_candidates.sort_values(['Segment', 'items'], ascending=[True, False])

        st.dataframe(boost_candidates[['Segment', 'productName', 'items', 'price', 'revenue']])

        # 8. Products to Reduce
        st.markdown('#### 📉 Products to Consider Reducing (Low Sales & Revenue)')
        st.markdown("> Items with low purchase frequency and low total revenue. Consider reducing stock or replacing with better options.")

        used_products = set(top5_popular['productName']) | set(bottom5_least['productName'])

        low_perf_products = (
            transaction_data_merged.groupby('productName')
            .agg({'items': 'sum', 'price': 'mean'})
            .reset_index()
        )
        low_perf_products['revenue'] = low_perf_products['items'] * low_perf_products['price']
        low_perf_products = low_perf_products[~low_perf_products['productName'].isin(used_products)]

        products_to_reduce = low_perf_products.sort_values(['items', 'revenue'], ascending=[True, True]).head(10)

        st.dataframe(products_to_reduce)

        # 9. Retention Product Candidates
        st.markdown("#### 🧲 Products with High Retention Potential")
        st.markdown("> Frequently re-purchased products, often by loyal customers. Useful for retention strategies and personalized offers.")

        product_freq_segment = (
            transaction_data_merged.groupby(['Segment', 'productName'])
            .agg({'items': 'sum', 'Member_number': pd.Series.nunique})
            .reset_index()
            .rename(columns={'items': 'total_items', 'Member_number': 'num_buyers'})
        )

        retention_segments = ['VIP', 'Normal']
        retention_products = product_freq_segment[product_freq_segment['Segment'].isin(retention_segments)].copy()
        retention_products['retention_score'] = retention_products['total_items'] * retention_products['num_buyers']

        top_retention_products = (
            retention_products.sort_values(['Segment', 'retention_score'], ascending=[True, False])
            .groupby('Segment')
            .head(5)
            .reset_index(drop=True)
        )

        st.dataframe(
            top_retention_products[['Segment', 'productName', 'total_items', 'num_buyers', 'retention_score']],
            use_container_width=True
        )

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
st.sidebar.markdown("**👩🏼‍🏫Supervisor:**")
st.sidebar.markdown("Ms. Khuat Thuy Phuong")  
st.sidebar.markdown("**🗓️Timeline:**")
st.sidebar.markdown("20/4/2025")
