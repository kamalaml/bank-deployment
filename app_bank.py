import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import datetime
import plotly.graph_objects as go
import joblib

# Load data
data = pd.read_csv('bank.csv', delimiter=',')


# Calculate Recency
data['last_contact_date'] = pd.to_datetime(data['day'].astype(str) + '-' + data['month'], format='%d-%b')
max_date = data['last_contact_date'].max()
data['recency'] = (max_date - data['last_contact_date']).dt.days

# Calculate Frequency
frequency = data.groupby('contact').size().reset_index(name='frequency')

# Calculate Monetary (using balance)
monetary = data.groupby('contact')['balance'].sum().reset_index(name='monetary')

# Merge RFM metrics
rfm = data[['contact', 'recency']].drop_duplicates().merge(frequency, on='contact').merge(monetary, on='contact')

# Define RFM score
rfm['R_score'] = pd.qcut(rfm['recency'], 4, labels=['1', '2', '3', '4'])
rfm['F_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 4, labels=['4', '3', '2', '1'])

# Determine unique bin edges and labels for monetary value
monetary_bins = pd.qcut(rfm['monetary'], 4, duplicates='drop')
monetary_labels = [str(i) for i in range(1, len(monetary_bins.cat.categories) + 1)]
rfm['M_score'] = pd.qcut(rfm['monetary'], len(monetary_labels), labels=monetary_labels)

# Combine RFM score
rfm['RFM_Score'] = rfm['R_score'].astype(str) + rfm['F_score'].astype(str) + rfm['M_score'].astype(str)

# Define customer segments based on RFM score
def segment_customer(row):
    if row['RFM_Score'] in ['444', '344', '434', '443']:
        return 'Best Customers'
    elif row['RFM_Score'] in ['244', '334', '324', '433', '343', '423', '414', '413', '341', '431']:
        return 'Loyal Customers'
    elif row['RFM_Score'] in ['122', '123', '133', '134', '124', '223', '224', '234']:
        return 'Potential Loyalist'
    elif row['RFM_Score'] in ['112', '113', '114', '123', '134', '142', '143']:
        return 'New Customers'
    elif row['RFM_Score'] in ['211', '212', '213', '214', '221', '222', '231', '232', '233', '241', '242', '243']:
        return 'Promising'
    elif row['RFM_Score'] in ['311', '312', '313', '314', '321', '322', '323', '331', '332', '333', '341', '342']:
        return 'Need Attention'
    elif row['RFM_Score'] in ['411', '421', '422', '431', '432']:
        return 'About to Sleep'
    elif row['RFM_Score'] in ['141', '142', '143', '144', '211', '212', '213', '214']:
        return 'At Risk'
    elif row['RFM_Score'] in ['111']:
        return 'Lost'
    else:
        return 'Others'

rfm['Customer_Segment'] = rfm.apply(segment_customer, axis=1)

# Merge the customer segments back to the original dataframe
data = data.merge(rfm[['contact', 'Customer_Segment']], on='contact', how='left')




# Convert 'last_contact_date' to datetime
data['last_contact_date'] = pd.to_datetime(data['last_contact_date'], format='%Y-%m-%d')

# Sidebar for filters
st.sidebar.title("Filters")
selected_job = st.sidebar.selectbox("Select Job", options=["All"] + list(data['job'].unique()))
selected_marital = st.sidebar.selectbox("Select Marital Status", options=["All"] + list(data['marital'].unique()))
selected_education = st.sidebar.selectbox("Select Education Level", options=["All"] + list(data['education'].unique()))
selected_segment = st.sidebar.selectbox("Select Customer Segment", options=["All"] + list(data['Customer_Segment'].unique()))

# Filter data based on selections
if selected_job != "All":
    data = data[data['job'] == selected_job]
if selected_marital != "All":
    data = data[data['marital'] == selected_marital]
if selected_education != "All":
    data = data[data['education'] == selected_education]
if selected_segment != "All":
    data = data[data['Customer_Segment'] == selected_segment]

# Page navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Job Analysis", "Marital Analysis & Education Analysis", "Time Analysis", "Customer Segment Analysis", "Prediction"])

# Page: Overview
if page == "Overview":
    st.markdown("<h1 style='color: #2E688C;'>Bank Marketing Analysis Dashboard - Overview</h1>", unsafe_allow_html=True)
    # Calculate metrics
    total_balance = data['balance'].sum()
    total_campaign_contacts = data['campaign'].sum()
    total_housing_loans = data[data['housing'] == 'yes'].shape[0]
    total_default_loans = data[data['default'] == 'yes'].shape[0]
    total_deposit_yes = data[data['deposit'] == 'yes'].shape[0]
    total_deposit_no = data[data['deposit'] == 'no'].shape[0]

    
    # Display metrics as styled cards
    st.markdown("""
        <style>
        .card {
            padding: 10px;
            margin: 5px;
            border-radius: 10px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
            text-align: center;
            background-color: #add8e6;
            font-weight: bold;
        }
        .card h2 {
            margin: 0;
            font-size: 16px;
        }
        .card p {
            margin: 0;
            font-size: 20px;
        }
        </style>
        """, unsafe_allow_html=True)

    # Create columns for layout
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f'<div class="card"><h2>Total Balance</h2><p>${total_balance:,.2f}</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="card"><h2>Total Campaign Contacts</h2><p>{total_campaign_contacts}</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="card"><h2>Total Housing Loans</h2><p>{total_housing_loans}</p></div>', unsafe_allow_html=True)

    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown(f'<div class="card"><h2>Total Default Loans</h2><p>{total_default_loans}</p></div>', unsafe_allow_html=True)
    with col5:
        st.markdown(f'<div class="card"><h2>Total Deposits (Yes)</h2><p>{total_deposit_yes}</p></div>', unsafe_allow_html=True)
    with col6:
        st.markdown(f'<div class="card"><h2>Total Deposits (No)</h2><p>{total_deposit_no}</p></div>', unsafe_allow_html=True)

    # Create columns for layout of pie charts
    col7, col8 = st.columns(2)
    col9, col10 = st.columns(2)

    # Define color sequence (shades of blue)
    colors = px.colors.sequential.Blues
    with col7:
        st.subheader('Education Distribution')
        education_distribution = data['education'].value_counts()
        fig_education = px.pie(education_distribution, names=education_distribution.index, values=education_distribution.values, title='Distribution by Education', width=400, height=400, color_discrete_sequence=colors)
        st.plotly_chart(fig_education)

    # Pie Chart for Job
    with col8:
        st.subheader('Job Distribution')
        job_distribution = data['job'].value_counts()
        fig_job = px.pie(job_distribution, names=job_distribution.index, values=job_distribution.values, title='Distribution by Job', width=400, height=400, color_discrete_sequence=colors)
        st.plotly_chart(fig_job)

    # Pie Chart for Marital Status
    with col9:
        st.subheader('Marital Status Distribution')
        marital_distribution = data['marital'].value_counts()
        fig_marital = px.pie(marital_distribution, names=marital_distribution.index, values=marital_distribution.values, title='Distribution by Marital Status', width=400, height=400, color_discrete_sequence=colors)
        st.plotly_chart(fig_marital)

    # Pie Chart for Poutcome
    with col10:
        st.subheader('Poutcome Distribution')
        poutcome_distribution = data['poutcome'].value_counts()
        fig_poutcome = px.pie(poutcome_distribution, names=poutcome_distribution.index, values=poutcome_distribution.values, title='Distribution by Poutcome', width=400, height=400, color_discrete_sequence=colors)
        st.plotly_chart(fig_poutcome)

    fig_age = px.histogram(data, x='age', nbins=30, title='Age Distribution of Clients',color_discrete_sequence=['#2E688C'])
    st.plotly_chart(fig_age)

    fig_balance = px.histogram(data, x='balance', nbins=30, title='Balance Distribution of Clients', color_discrete_sequence=['#2E688C'])
    st.plotly_chart(fig_balance)
    
# Page: Job Analysis
elif page == "Job Analysis":
    st.markdown("<h1 style='color: #2E688C;'>Bank Marketing Analysis Dashboard - Job Analysis</h1>", unsafe_allow_html=True)
    st.subheader('Distribution of Job Types')
    job_counts = data['job'].value_counts()
    st.bar_chart(job_counts)
    
    job_campaign_counts = data.groupby('job')['campaign'].sum().reset_index().sort_values(by = 'campaign',ascending = False)
    # Create a funnel chart
    fig_funnel = px.funnel(job_campaign_counts, x='campaign', y='job', title='Campaign Contacts Funnel by Job Type',color_discrete_sequence=['#add8e6'] ,labels={'job': 'Job Type', 'campaign': 'Number of Campaign Contacts'})
    st.plotly_chart(fig_funnel)

    colors = px.colors.sequential.Blues
    st.subheader('Deposit by Job')
    job_deposit = data.groupby('job')['deposit'].value_counts().unstack().fillna(0)
    fig_job = px.pie(job_deposit, names=job_deposit.index, values='yes', title='Deposits by Job', width=400, height=400, color_discrete_sequence=colors)
    st.plotly_chart(fig_job)
    
    
# Page: Marital Analysis
elif page == "Marital Analysis & Education Analysis":
    colors = px.colors.sequential.Blues
    st.markdown("<h1 style='color:#2E688C;'>Bank Marketing Analysis Dashboard - Education Analysis</h1>", unsafe_allow_html=True)
    grouped_df = data.groupby('marital')[['housing']].count().reset_index()
    fig = px.bar(grouped_df, x='marital', y='housing',color_discrete_sequence=['#2E688C'], 
             title='Marital Status vs. Housing loan',
             labels={'marital': 'Marital Status', 'housing': 'Count'})

    st.plotly_chart(fig)
    
    fig_marital = px.histogram(data, x='marital', color='deposit', title='Marital Status vs. Deposit Subscription', 
                           barmode='group', color_discrete_sequence=colors)
    st.plotly_chart(fig_marital)
    
    st.subheader('Balance Distribution by Education Level')

    # Calculate the average balance for each education level
    education_balance = data.groupby('education')['balance'].mean().reset_index()

    # Create the bar chart
    fig3 = px.bar(education_balance, x='education', y='balance', 
                  title='Balance Distribution by Education Level',
                  labels={'education': 'Education Level', 'balance': 'Average Balance'},
                  color='education', 
                  color_discrete_sequence=colors)

    # Show the chart in Streamlit
    st.plotly_chart(fig3)
    
    fig_education = px.histogram(data, x='education', color='deposit', title='Education Level vs. Deposit Subscription', 
                              barmode='group', color_discrete_sequence=colors)
    fig_education.update_xaxes(categoryorder='total descending')
    st.plotly_chart(fig_education)
    
# Page: Education Analysis
elif page == "Time Analysis":
    st.markdown("<h1 style='color:#2E688C;'>Bank Marketing Analysis Dashboard</h1>", unsafe_allow_html=True)
    month_campaign_counts = data.groupby('month')['campaign'].sum().reset_index()
    fig_month = px.bar(month_campaign_counts, x='month', y='campaign', title='Campaign Contacts Funnel by Month',color_discrete_sequence=['#1f77b4'], labels={'month': 'Month', 'campaign': 'Number of Campaign Contacts'})
    st.plotly_chart(fig_month)
    
    data['duration_min'] = data['duration'] / 60

    # Group the data by day and sum the campaign contacts and duration (in minutes) for each day
    day_campaign_duration = data.groupby('day').agg({'campaign': 'sum', 'duration_min': 'sum'}).reset_index()

    # Create the line chart
    fig_line = go.Figure()

    fig_line.add_trace(go.Scatter(x=day_campaign_duration['day'], y=day_campaign_duration['campaign'], mode='lines+markers', 
                               name='Campaign Contacts', yaxis='y1', line=dict(color='#1f77b4')))

    fig_line.add_trace(go.Scatter(x=day_campaign_duration['day'], y=day_campaign_duration['duration_min'], mode='lines+markers', 
                                   name='Duration (min)', yaxis='y2', line=dict(color='#aec7e8')))

    # Update the layout
    fig_line.update_layout(
        title='Campaign Contacts and Duration by Day',
        xaxis=dict(title='Day'),
        yaxis=dict(
            title='Campaign Contacts',
            titlefont=dict(color='#1f77b4'),
            tickfont=dict(color='#1f77b4')
        ),
        yaxis2=dict(
            title='Duration (min)',
            titlefont=dict(color='#ff7f0e'),
            tickfont=dict(color='#ff7f0e'),
            overlaying='y',
            side='right'
        ),
        legend=dict(
            x=0.1,
            y=1.1,
            orientation='h'
        )
    )
    st.plotly_chart(fig_line)
    
    agg_data = data.groupby('deposit')['pdays'].sum().reset_index()
    st.title('Sum of Days Since Last Contact by Deposit Status')
    fig_deposit = px.bar(agg_data, x='deposit', y='pdays', color='deposit',
                                   color_discrete_sequence=['#1f77b4', '#aec7e8'],
                                   title='Sum of Days Since Last Contact by Deposit Status',
                                   labels={'pdays': 'Sum of Days Since Last Contact', 'deposit': 'Deposit Status'})
    st.plotly_chart(fig_deposit)
    
    agg_data = data.groupby('poutcome')['previous'].sum().reset_index()
    st.title('Sum of Previous Contacts by Poutcome')
    fig_previous_poutcome = px.bar(agg_data, x='poutcome', y='previous', color_discrete_sequence=['#1f77b4'],
                                   title='Sum of Previous Contacts by Poutcome',
                                   labels={'previous': 'Sum of Previous Contacts', 'poutcome': 'Poutcome'})
    st.plotly_chart(fig_previous_poutcome)



# Page: Customer Segment Analysis
elif page == "Customer Segment Analysis":
    st.markdown("<h1 style='color:#2E688C;'>Bank Marketing Analysis Dashboard - Customer Segment Analysis</h1>", unsafe_allow_html=True)
    st.subheader('Distribution of Customer Segments')
    segment_counts = data['Customer_Segment'].value_counts()
    st.bar_chart(segment_counts)

    st.subheader('Recency Distribution')
    fig6, ax6 = plt.subplots()
    ax6.hist(data['recency'], bins=20, edgecolor='black')
    ax6.set_xlabel('Recency (days)')
    ax6.set_ylabel('Count')
    st.plotly_chart(fig6)
    
elif page=="Prediction":
    model = joblib.load('model.pkl')

    def predict_investment(age,job,marital,education,default,balance,housing,loan,contact,day,month,duration,campaign,pdays,previous,poutcome):
        prediction = model.predict(pd.DataFrame({"age": [age] , "job" : [job] , 'marital': [marital] , 'education' : [education] ,\
                                               'default' : [default], "balance":[balance], "housing" :[housing], "loan" :[loan],\
                                               "contact" : [contact],"day" : [day], "month":[month], "duration" :[duration],'campaign' : [campaign], "pdays":[pdays], "previous" :[previous], "poutcome" :[poutcome]}))
        return prediction

    def main():
        st.title('investment prediction')
        html_temp="""
                    <div style="background-color:#CD9E8E">
                    <h2 style="color:white;text-align:center;">Bank Marketing</h2>
                    </div>
                  """
        st.markdown(html_temp,unsafe_allow_html=True)

        age = st.text_input('Age')
        job = st.radio('pick your Job',['admin.', 'technician', 'services', 'management', 'retired','blue-collar', 'unemployed', 'entrepreneur', 'housemaid','unknown', 'self-employed', 'student'] )
        marital = st.radio('Marital Status',['married', 'single', 'divorced'])
        education = st.radio('Education', ['secondary', 'tertiary', 'primary', 'unknown'])
        default = st.radio('Are you have credit card?' ,['no', 'yes'] )
        balance = st.text_input('Balance')
        housing = st.radio('Housing loan',['yes', 'no'])
        loan = st.radio('Personal Loan',['no', 'yes'])
        contact = st.radio('Contact Type',['unknown', 'cellular', 'telephone'] )
        day = st.text_input('day')
        month = st.radio('Contact Type',['may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'jan', 'feb','mar', 'apr', 'sep'] )
        duration = st.text_input('duration' )
        campaign = st.text_input('Campaign')
        pdays = st.text_input('Pdays')
        previous = st.text_input('Previous' )
        poutcome = st.radio('Poutcome', ['unknown', 'other', 'failure', 'success'])

        result = ""

        if st.button('predict'):
            result = predict_investment(age,job,marital,education,default,balance,housing,loan,contact,day,month,duration,campaign,pdays,previous,poutcome)

            st.success('this person {}'.format(result))

    if __name__ =='__main__':
        main()



    
