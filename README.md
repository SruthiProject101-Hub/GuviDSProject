# GuviDSProject
This repository includes all Projects completed as part of DS Course
DS Mini Project 1 Documentation: Analyzing Personal Expenses
# 1. Project Overview
The Personal Expense Tracker app is designed to analyse and categorize an individual’s spending habits across different categories and payment modes to plan actionable savings. This app connects to an SQLite database (expenses.db) containing historical transaction data and provides various queries to analyze and visualize the data. The app provides insights into the user’s spending behavior, helping to identify areas where they can save or optimize their spending.
# 2. Libraries Used
This project uses a variety of Python libraries to simulate data, store it in a SQL database, perform data analysis, and build an interactive web application for visualization. The primary libraries include:

- Faker: To simulate realistic transaction data such as dates, descriptions, and categories.
- Pandas: For handling and manipulating tabular data efficiently.
- Numpy and Random: To generate random numbers and support numeric operations.
- Sqlite3: For managing a lightweight local database to store expense records.
- Matplotlib and Seaborn: For plotting visualizations such as bar graphs, pie charts, and line plots in Jupyter or Colab.
- Streamlit: To develop an interactive web-based dashboard to explore the expense data using SQL queries and visualizations.
# 3. Data Simulation
I utilized the Faker library along with Python’s built-in random utilities to generate a synthetic dataset representing a full year of an individual's personal expenses. Each record includes fields such as the transaction date, category (e.g., groceries, bills, subscriptions), payment mode (cash, online), description, amount paid, and cashback (if any). Approximately 1200 entries were created with randomized yet realistic distributions for each field.
# 4. SQL Table Creation and Data Insertion
Using the sqlite3 library, I created a database file named 'expenses.db' to store the simulated expense records. The data was inserted into a single table named 'expenses' with a schema including id, date, category, payment mode, description, amount_paid, and cashback. The data from the pandas DataFrame was inserted using the to_sql() method, ensuring seamless integration between the simulated data and the SQL database.
# 5. Querying and Visualization in Google Colab
A total of 15 SQL queries were written to extract meaningful insights from the expense data. These queries answered questions about total spending by category or payment mode, cashback earned, monthly trends, recurring expenses, and high-priority spending. In Google Colab, these queries were executed using a helper function run_query() that connects to the database, executes the SQL, and returns the result as a pandas DataFrame. The query results were visualized using Matplotlib and Seaborn with appropriate chart types including bar graphs, pie charts, and line plots for better understanding of patterns.
# 6. Streamlit App Development
After data simulation and initial analysis in Colab, I transitioned to building an interactive dashboard using Streamlit. The Streamlit app was developed locally in a project folder containing the expenses.db file and the app.py script. Streamlit widgets like selectbox allowed to choose any of the 15 predefined queries. The application used the same run_query() function to retrieve data, display it in tabular form using st.dataframe(), and render dynamic visualizations based on user selection. Charts were built with Matplotlib and Seaborn and displayed in Streamlit using st.pyplot().
# 7. Running the Application Locally
To run the Streamlit application, necessary libraries such as pandas, matplotlib, seaborn, and streamlit were installed using pip. The terminal was used to navigate to the project directory, and the app was launched using the command: streamlit run app.py. This opened a web interface that provided users with query selection, output display, and associated visual insights, all powered by a local SQL database.
## Project Deliverables
1. ●	Source code for data cleaning, SQL integration, EDA, and Streamlit app.
[Expense_Tracker_Documentation - Final - Sruthi.V.S.docx](https://github.com/user-attachments/files/20068682/Expense_Tracker_Documentation.-.Final.-.Sruthi.V.S.docx)
2. ●	SQL scripts for all 20 queries.
   [15 SQL queries.txt](https://github.com/user-attachments/files/20068577/15.SQL.queries.txt)
3. ●	Documentation explaining the methodology, analysis, and insights.
   [Expense_Tracker_Documentation - Final - Sruthi.V.S.docx](https://github.com/user-attachments/files/20068592/Expense_Tracker_Documentation.-.Final.-.Sruthi.V.S.docx)
4. ●	Screenshots of the Streamlit app with key visualizations and outputs.
   [15 SQL Query Visualizations from Streamlit App.docx](https://github.com/user-attachments/files/20068672/15.SQL.Query.Visualizations.from.Streamlit.App.docx)





