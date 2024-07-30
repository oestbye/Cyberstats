import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import pygsheets
from google.oauth2 import service_account
import numpy as np

DEBUG_MODE = True

# Function to print debug statements
def debug_print(message):
    if DEBUG_MODE:
        print(message)

# Set page configuration and title
st.set_page_config(
    page_title="Cyber Security Market Analysis, Norway",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

short_bio = """
    I embarked on this project to enhance my coding skills and due to my fascination 
    with analyzing annual financial figures of various companies. By automating this 
    process and integrating graphical representations, I aim to inject a layer of 
    enjoyment into the analysis—not just for myself, but potentially for you too. 
    Feel free to contact me if you have any questions or requests.
"""

# Split into two columns
col1, col2 = st.columns([4, 2])

with col1:
    st.title("Cyber Security Market Analysis, Norway")
    st.header("About")
    st.markdown("""
        Welcome to this analysis of the Norwegian Cyber Security market.
        This section features graphs that illustrate market growth, market share distribution among companies, detailed financial summaries of these companies, trends, and more.
    """)
    st.header("Scope")
    st.markdown("""
        I have selected only those companies with at least 60% of their income derived from cyber security.
        This ensures the analysis is as accurate as possible. Including data from the big consultancy companies would be an issue as i dont have any insight into what parts of the income and result are connected to cyber security services.
        Keep this in mind when you look at the numbers and the graphs. Ping me if you know of a company that qualifies and that should be added.
        The 2023 graphs will be populated when most of the 2023 data is available.
    """)
    st.header("Technology and Data Fetching")
    st.markdown("""
        Data is fetched from the Brønnøysundregisteret API and my experience so far is that the data is available short time after the financial statements are approved by the accounting register, and is usually published here 1 day before proff.no
        To trigger a data fetch from the API, please navigate to the bottom of the page and click the "check for new data" button. A function will then run to check if there is any new data available in the API.
        The technology I've used to build this web application includes Streamlit, Pandas, Plotly Express, and Google Sheets. Everything is written in Python.
        The Python code is open-sourced if you're curious and wish to take a look. Please remember that it was not written by a professional developer and will not be maintained.
        The primary reason for sharing is to lower the threshold for open-sourcing code, even among those who do not pursue this as a profession.
    """)
    st.header("Looking for Cyber Security Consultancy, Services, or Software?")
    st.markdown("""
        If you are interested in Cloud Security or have Cloud Security challenges you would like to discuss, 
        please feel free to connect with me on LinkedIn and/or send me an email using the links below my picture. 
        O3 Cyber (O3C) is a specialized Cloud Security Consultancy Boutique, 
        dedicated to providing expert solutions tailored to your needs.

        At the bottom of this page, you will also find a comprehensive table listing companies that meet the specified scope. 
        This table includes detailed information about the services they offer, assisting you in finding the right company for your cyber security needs.
            """)

linkedin_url = "https://www.linkedin.com/in/oestbye/"
O3C_link = "https://www.o3c.io"
github_url = "https://github.com/oestbye/Cyberstats"
image_url = 'https://iili.io/J1miME7.png'
O3C_logo = 'https://iili.io/duuPnCx.md.png'

with col2:
    st.image(image_url, caption='', use_column_width=True)
    st.header("Olav Østbye")
    st.write("Principal Cloud Security Manager @ O3 Cyber (O3C)")
    st.caption(short_bio)

    # Arrange buttons in a row using columns with smaller width ratios
    button_col1, button_col2, button_col3 = st.columns([1, 1, 1])
    with button_col1:
        st.link_button("Linkedin", url=linkedin_url)
    with button_col2:
        st.link_button("Email", url="mailto:olav@o3c.no")
    with button_col3:
        st.link_button("GitHub", url=github_url)
    
    st.markdown(f'<div style="text-align: center;"><a href="{O3C_link}" target="_blank"><img src="{O3C_logo}" alt="O3C Logo" style="max-width: 75%;"></a>', unsafe_allow_html=True)

st.markdown("---")

# Function to get financial data from the BRREG API
def get_financial_data(org_number):
    url = f"https://data.brreg.no/regnskapsregisteret/regnskap/{org_number}?regnskapstype=SELSKAP"
    try:
        debug_print(f"Fetching data for org number: {org_number} from URL: {url}")
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError if the response status is 4xx or 5xx
        data = response.json()
        debug_print(f"Data fetched successfully for org number: {org_number}")
        return data
    except requests.exceptions.HTTPError as e:
        error_message = f"Error fetching data for org number {org_number}: {e}"
        if response.status_code == 404:
            error_message += " This may be because the company is new and/or data is not yet available."
        debug_print(error_message)
        return None
    except requests.RequestException as e:
        error_message = f"Error fetching data for org number {org_number}: {e}"
        debug_print(error_message)
        return None


def convert_columns_to_float(df):
    numeric_columns = [
        'Income 2023', 'Result 2023', 'Profit margin 2023', 'Market share 2023', 
        'Income 2022', 'Result 2022', 'Profit margin 2022', 'Market share 2022', 
        'Income 2021', 'Result 2021', 'Profit margin 2021', 'Market share 2021', 
        'Income 2020', 'Result 2020', 'Profit margin 2020', 'Market share 2020'
    ]

    for column in numeric_columns:
        try:
            # Replace specific string patterns with NaN and convert to float
            df[column] = (
                df[column]
                .astype(str)  # Ensure the column is treated as string to apply string methods
                .str.replace('\xa0', '')  # Remove non-breaking spaces
                .str.replace(',', '.')  # Replace commas with periods for decimal points
                .str.replace('−', '-')  # Standardize negative signs
                .replace('#DIV/0!', np.nan)  # Replace division errors with NaN
                .replace('None', np.nan)  # Replace 'None' strings with NaN
                .replace('', np.nan)  # Replace empty strings with NaN
                .astype(float, errors='ignore')  # Attempt to convert to float, ignore errors if conversion fails
            )
            df[column] = df[column].infer_objects()
        except Exception as e:
            debug_print(f"Error processing column '{column}': {e}")
    return df

def format_financial_and_percentage_columns(df):
    currency_cols = ['Income 2023', 'Result 2023', 'Income 2022', 'Result 2022',
                     'Income 2021', 'Result 2021', 'Income 2020', 'Result 2020']
    percent_cols = ['Market share 2023', 'Market share 2022', 'Market share 2021',
                    'Market share 2020', 'Profit margin 2023', 'Profit margin 2022',
                    'Profit margin 2021', 'Profit margin 2020', 'Market Share Trend 2020-2022']

    # Format currency columns
    for col in currency_cols:
        if col in df.columns:
            # Convert to numeric and handle NaNs
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Round to one decimal place
            df[col] = df[col].round(1)

    # Format percent columns
    for col in percent_cols:
        if col in df.columns:
            # Convert string representation of numbers with commas as decimal separators to float
            # and replace en-dash with hyphen-minus for negative numbers
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(',', '.')  # Replace comma with dot for decimal
                .str.replace('−', '-')  # Replace en-dash/other dashes with hyphen-minus for negative numbers
                .astype(float, errors='ignore')  # Convert string to float, ignore errors
            )
            df[col] = (df[col] * 100).round(2)  # Round to two decimal places for percentage
    return df


def fetch_existing_df():
    try:
        # Open the Google Sheet and select the 'selskapsdata' worksheet
        sh = gc.open("stats.xlsx")
        selskapsdata_worksheet = sh.worksheet('title', 'selskapsdata')
        # Get the 'selskapsdata' data as DataFrame
        selskapsdata_df = selskapsdata_worksheet.get_as_df()
        return selskapsdata_df
    except Exception as e:
        st.error(f"Error fetching 'selskapsdata' data: {e}")
        return None

def fetch_other_df():
    try:
        # Open the Google Sheet and select the 'other' worksheet
        sh = gc.open("stats.xlsx")
        other_worksheet = sh.worksheet('title', 'other')

        other_df = other_worksheet.get_as_df(has_header=False, numerize=False).iloc[:1]
        
        return other_df
    except Exception as e:
        st.error(f"Error fetching 'other' data: {e}")
        return None

def sanitize_number(number_str):
    if number_str is not None and isinstance(number_str, str):
        number_str = number_str.strip()
        number_str = number_str.replace('\xa0', '')  # Remove non-breaking space
        number_str = number_str.replace('.', '')  # Remove period (thousand separator in European format)
        number_str = number_str.replace(',', '.')  # Change comma to period for decimal
        number_str = number_str.replace('−', '-')  # Replace en dash with hyphen-minus
        try:
            return float(number_str)
        except ValueError:
            return None
    return number_str

def get_current_values_from_sheet(worksheet, org_number, year):
    try:
        sh = gc.open("stats.xlsx")
        worksheet = sh.sheet1
        cells = worksheet.find(str(org_number))
        if cells:
            org_cell = cells[0]
            row_number = org_cell.row

            income_col_cells = worksheet.find(f"Income {year}")
            result_col_cells = worksheet.find(f"Result {year}")

            current_income = None
            current_result = None

            if income_col_cells:
                income_col_number = income_col_cells[0].col
                current_income = worksheet.cell((row_number, income_col_number)).value
                debug_print(f"Current income from sheet: {current_income}")
            else:
                debug_print(f"Income column for year {year} not found.")

            if result_col_cells:
                result_col_number = result_col_cells[0].col
                current_result = worksheet.cell((row_number, result_col_number)).value
                debug_print(f"Current result from sheet: {current_result}")
            else:
                debug_print(f"Result column for year {year} not found.")

            return sanitize_number(current_income), sanitize_number(current_result)
        else:
            debug_print(f"No matching cells found for org number {org_number}")
            return None, None
    except Exception as e:
        debug_print(f"Error fetching current values from sheet for org number {org_number}: {e}")
        return None, None

def check_and_update_sheet(worksheet, org_number, year, new_income, new_result, existing_income, existing_result):
    sh = gc.open("stats.xlsx")
    worksheet = sh.sheet1
    # Initialize update flags
    income_needs_update = False
    result_needs_update = False

    # Check if new_income and existing_income are not None before comparing
    if new_income is not None:
        income_needs_update = existing_income is None or abs(new_income - existing_income) / max(abs(existing_income), 1) > TOLERANCE_PERCENT

    # Check if new_result and existing_result are not None before comparing
    if new_result is not None:
        result_needs_update = existing_result is None or abs(new_result - existing_result) / max(abs(existing_result), 1) > TOLERANCE_PERCENT

    # Update the worksheet if needed
    if income_needs_update:
        update_cell(worksheet, org_number, year, 'Income', new_income)
    else:
        debug_print(f"No update needed for Income for {org_number} for the year {year}")

    if result_needs_update:
        update_cell(worksheet, org_number, year, 'Result', new_result)
    else:
        debug_print(f"No update needed for Result for {org_number} for the year {year}")

def update_cell(worksheet, org_number, year, field, new_value):
    try:
        debug_print(f"Attempting to update {field} for org number: {org_number}, year: {year}")

        # Find the row for the org number
        sh = gc.open("stats.xlsx")
        worksheet = sh.sheet1
        org_cells = worksheet.find(str(org_number))
        if not org_cells:
            debug_print(f"Organization number {org_number} not found in the sheet.")
            return

        # Find the column for the year and field
        year_field = f"{field} {year}"
        year_cells = worksheet.find(year_field)
        if not year_cells:
            debug_print(f"{year_field} column not found in the sheet.")
            return

        row_index = org_cells[0].row
        col_index = year_cells[0].col

        # Update the cell value
        worksheet.update_value((row_index, col_index), new_value)
        debug_print(f"Updated {field} for {org_number} in {year} to {new_value}")
    except Exception as e:
        debug_print(f"Error updating cell for org number {org_number} in {year}: {e}")

def is_2023_data_available(df):
    # Clean the 'Organization Number' column
    df['Organization Number'] = df['Organization Number'].astype(str).str.strip()

    # Debug print the cleaned dataframe
    debug_print(f"Cleaned dataframe:\n{df}")

    # Define the org numbers of the 3 biggest companies
    big_companies = ['982089549', '993856886', '968504436', '981548280']

    # Debug print the relevant rows for each company
    for org_number in big_companies:
        relevant_rows = df[df['Organization Number'] == org_number]
        debug_print(f"Relevant rows for company {org_number}:\n{relevant_rows}")

        is_data_available = relevant_rows['Income 2023'].notna().any()
        debug_print(f"Data for company {org_number} available: {is_data_available}")

    # Check if the data for these companies for 2023 is available
    big_companies_data_available = all(
        df[df['Organization Number'] == org_number]['Income 2023'].notna().any()
        for org_number in big_companies
    )

    # Count the total number of companies with available 2023 data
    total_companies_with_2023_data = df['Income 2023'].notna().sum()

    # Debug print the total number of companies with available 2023 data
    debug_print(f"Total companies with 2023 data: {total_companies_with_2023_data}")

    # Return True if conditions are met, else False
    result = big_companies_data_available and total_companies_with_2023_data > 10
    debug_print(f"Result of data availability check: {result}")

    return result


service_account_info = {
    "type": st.secrets["pygsheets"]["type"],
    "project_id": st.secrets["pygsheets"]["project_id"],
    "private_key_id": st.secrets["pygsheets"]["private_key_id"],
    "private_key": st.secrets["pygsheets"]["private_key"].replace('\\n', '\n'),
    "client_email": st.secrets["pygsheets"]["client_email"],
    "client_id": st.secrets["pygsheets"]["client_id"],
    "auth_uri": st.secrets["pygsheets"]["auth_uri"],
    "token_uri": st.secrets["pygsheets"]["token_uri"],
    "auth_provider_x509_cert_url": st.secrets["pygsheets"]["auth_provider_x509_cert_url"],
    "client_x509_cert_url": st.secrets["pygsheets"]["client_x509_cert_url"]
}

credentials = service_account.Credentials.from_service_account_info(
    service_account_info,
    scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
)

# Authorize pygsheets with the credentials
gc = pygsheets.authorize(custom_credentials=credentials)

try:
    # Attempt to open the spreadsheet
    debug_print("Attempting to open the spreadsheet 'stats.xlsx'")
    sh = gc.open("stats.xlsx")
    worksheet = sh.sheet1
    debug_print("Spreadsheet 'stats.xlsx' opened successfully")
except Exception as e:
    error_message = f"Error while accessing spreadsheet: {e}"
    debug_print(error_message)

def get_and_update_last_updated_time():
    # Initialize 'last_updated_time' in st.session_state if it doesn't exist
    if 'last_updated_time' not in st.session_state:
        st.session_state['last_updated_time'] = datetime.now() - timedelta(hours=1)
    
    try:
        # Fetch the 'other' DataFrame
        other_df = fetch_other_df()

        # Debug print the entire DataFrame to check its structure and contents
        debug_print(f"DataFrame contents:\n{other_df}")

        # Check if the DataFrame is not empty
        if other_df is not None and not other_df.empty:
            # Assuming that the 'Last updated:' label is in the first column and the timestamp in the second
            last_updated_str = other_df.iloc[0, 1]
            debug_print(f"Retrieved 'Last updated:' from DataFrame: {last_updated_str}")

            # Convert the last updated string to datetime
            last_updated_time = pd.to_datetime(last_updated_str, errors='coerce')
            if not pd.isnull(last_updated_time):
                st.session_state['last_updated_time'] = last_updated_time
                debug_print(f"Session state updated with last updated time: {last_updated_time}")
            else:
                raise ValueError(f"Unable to parse 'Last updated' time: {last_updated_str}")
        else:
            # If the DataFrame is empty or the expected data is not found, log a message
            debug_print("'Last updated:' not found or DataFrame is empty. No update made to the session state.")
            
    except Exception as e:
        st.error(f"Error handling the last updated time: {e}")
        debug_print(f"Exception: {e}")

def update_last_update_time_in_sheet(sheet_name, cell, current_time_str):
    try:
        # Debug print: Attempting to update the sheet
        debug_print(f"Attempting to update sheet '{sheet_name}' at cell '{cell}' with time '{current_time_str}'")

        # Open the specific worksheet by title
        other_worksheet = sh.worksheet('title', sheet_name)
        
        # Update the cell with the current time string
        other_worksheet.update_value(cell, current_time_str)
        
        debug_print(f"Successfully updated the sheet with last updated time: {current_time_str}")

        # Update the session state with the new last updated time
        st.session_state['last_updated_time'] = pd.to_datetime(current_time_str)
        
        debug_print(f"Session state updated with last updated time: {current_time_str}")

        st.experimental_rerun()
    except Exception as e:
        debug_print(f"Error updating 'Last updated' time in Google Sheet: {e}")

update_interval = timedelta(minutes=20)

def is_button_disabled():
    # Check if 'last_updated_time' is in the session state
    if 'last_updated_time' in st.session_state:
        # Convert last updated time from session state to datetime if it's a string
        last_updated_time = st.session_state['last_updated_time']
        if isinstance(last_updated_time, str):
            # Parse the string to a datetime object
            last_updated_time = datetime.strptime(last_updated_time, '%Y-%m-%d %H:%M:%S')
        
        # Calculate the elapsed time
        elapsed_time = datetime.now() - last_updated_time
        return elapsed_time < update_interval
    else:
        # If 'last_updated_time' is not set, do not disable the button
        return False

# Check if the button should be disabled
button_disabled = is_button_disabled()

# If the data difference is less than 2%, don't update
TOLERANCE_PERCENT = 0.02

# Check if new data is available in the API and not already in the google sheet
# First checking against the dataframe to save complexity, compute and time. If the difference is bigger than the tolerance_percent, then check the google sheet and update if necessary.
def update_google_sheet(worksheet, org_number, financial_data, existing_df):
    update_summary = []  # List to store updates for each organization number

    if financial_data:
        for item in financial_data:
            fraDato = item.get('regnskapsperiode', {}).get('fraDato', '')
            accounting_year = fraDato.split('-')[0] if fraDato else 'Unknown Year'

            new_income = item.get('resultatregnskapResultat', {}).get('driftsresultat', {}).get('driftsinntekter', {}).get('sumDriftsinntekter', None)
            new_result = item.get('resultatregnskapResultat', {}).get('ordinaertResultatFoerSkattekostnad', None)

            existing_df['Organization Number'] = existing_df['Organization Number'].astype(str).str.strip()
            org_number_str = str(org_number).strip()

            df_row = existing_df[existing_df['Organization Number'] == org_number_str]
            if not df_row.empty:
                existing_income_df = sanitize_number(df_row[f'Income {accounting_year}'].values[0] or 0)
                existing_result_df = sanitize_number(df_row[f'Result {accounting_year}'].values[0] or 0)

                # Check for updates against DataFrame data
                if (new_income is not None and (existing_income_df is None or abs(new_income - existing_income_df) / max(abs(existing_income_df), 1) > TOLERANCE_PERCENT)) or \
                   (new_result is not None and (existing_result_df is None or abs(new_result - existing_result_df) / max(abs(existing_result_df), 1) > TOLERANCE_PERCENT)):
                    # Fetch current values from the sheet for comparison
                    existing_income_sheet, existing_result_sheet = get_current_values_from_sheet(worksheet, org_number_str, accounting_year)

                    # Check for updates against Sheet data
                    if (new_income is not None and (existing_income_sheet is None or abs(new_income - existing_income_sheet) / max(abs(existing_income_sheet), 1) > TOLERANCE_PERCENT)) or \
                       (new_result is not None and (existing_result_sheet is None or abs(new_result - existing_result_sheet) / max(abs(existing_result_sheet), 1) > TOLERANCE_PERCENT)):
                        # Record updates
                        update_summary.append({
                            'org_number': org_number_str,
                            'accounting_year': accounting_year,
                            'income': {
                                'old': existing_income_sheet,
                                'new': new_income
                            },
                            'result': {
                                'old': existing_result_sheet,
                                'new': new_result
                            }
                        })
                        # Perform updates on the sheet
                        update_cell(worksheet, org_number_str, accounting_year, 'Income', new_income)
                        update_cell(worksheet, org_number_str, accounting_year, 'Result', new_result)
                else:
                    pass
            else:
                pass

    return update_summary

def get_column_letter(column_index):
    """Convert a column index into a column letter."""
    column_letter = ''
    while column_index > 0:
        column_index, remainder = divmod(column_index - 1, 26)
        column_letter = chr(65 + remainder) + column_letter
    return column_letter

try:
    # Open the spreadsheet by its title
    sh = gc.open("stats.xlsx")
    worksheet = sh.sheet1

    last_row = len(worksheet.get_col(1, include_tailing_empty=False))
    last_col = len(worksheet.get_row(1, include_tailing_empty=False))

    # Get the data as DataFrame
    df = worksheet.get_as_df(start='A1', num_rows=last_row, num_cols=last_col)

    # Convert all columns that can be converted to floats
    df = convert_columns_to_float(df)

    # Format financial and percentage columns
    df = format_financial_and_percentage_columns(df)

except Exception as e:
    st.error(f"Error fetching data: {e}")

# Column configurations with appropriate formats
column_configurations = {
    "Income 2023": st.column_config.NumberColumn("Income 2023"),
    "Result 2023": st.column_config.NumberColumn("Result 2023"),
    "Profit margin 2023": st.column_config.NumberColumn("Profit Margin 2023", format="%.2f%%"),
    "Market share 2023": st.column_config.NumberColumn("Market share 2023", format="%.2f%%"),
    "Income 2022": st.column_config.NumberColumn("Income 2022"),
    "Result 2022": st.column_config.NumberColumn("Result 2022"),
    "Profit margin 2022": st.column_config.NumberColumn("Profit Margin 2022", format="%.2f%%"),
    "Market share 2022": st.column_config.NumberColumn("Market share 2022", format="%.2f%%"),
    "Income 2021": st.column_config.NumberColumn("Income 2021"),
    "Result 2021": st.column_config.NumberColumn("Result 2021"),
    "Profit margin 2021": st.column_config.NumberColumn("Profit Margin 2021", format="%.2f%%"),
    "Market share 2021": st.column_config.NumberColumn("Market share 2021", format="%.2f%%"),
    "Income 2020": st.column_config.NumberColumn("Income 2020"),
    "Result 2020": st.column_config.NumberColumn("Result 2020"),
    "Profit margin 2020": st.column_config.NumberColumn("Profit Margin 2020", format="%.2f%%"),
    "Market share 2020": st.column_config.NumberColumn("Market share 2020", format="%.2f%%"),
    "Market Share Trend 2020-2022": st.column_config.NumberColumn("Market Share Trend 2020-2022", format="%.2f%%"),
}

# Define main_table_columns before using it
main_table_columns = [col for col in df.columns if col not in ['Type', 'Company type', 'Specialization', 'Proff URL', 'Website URL']]

include_2023_data = is_2023_data_available(df)
if DEBUG_MODE:
    print(f"Is enough 2023 data available? {'Yes' if include_2023_data else 'No'}")

# Display the DataFrame
st.title("Financial Data per company 2020-2023")

# Explanatory text for the Financial Data table
st.markdown("""
The table below offers a detailed snapshot of the financial metrics for entities within Norway's Cyber Security sector, spanning from 2020 to 2023. 
It encompasses annual data on revenue, results, profit margins, and market shares for each listed company, and more.
Interpreting data from a table can sometimes be challenging. For a more intuitive understanding, please check out the graphs below the table, which visually represent the data for clearer insights.

Please note that the data for 2023 is updated as it becomes available.
""")

# Remove the columns you don't want to display
main_table_columns = [col for col in df.columns if col not in ['Type', 'Company type', 'Specialization', 'Proff URL', 'Website URL']]

st.data_editor(data=df[main_table_columns], column_config=column_configurations)

st.markdown("""
### Growth of Specialized Cyber Security Companies Based on Year
""")

# Create a copy of the DataFrame for the specific analysis
df_established = df.copy()

# Convert the 'Established' column to numeric, handling errors, in the copy
df_established['Established'] = pd.to_numeric(df_established['Established'], errors='coerce')

# Drop rows with NaN values in 'Established' column (if any), in the copy
df_established.dropna(subset=['Established'], inplace=True)

# Ensure the 'Established' column is of integer type for proper sorting and representation
df_established['Established'] = df_established['Established'].astype(int)

# Count the number of companies established each year
companies_per_year = df_established['Established'].value_counts().sort_index()

# Calculate the cumulative sum to represent growth over the years
cumulative_companies = companies_per_year.cumsum()

# Create a new DataFrame for plotting
growth_df = pd.DataFrame({
    'Year': cumulative_companies.index,
    'Cumulative Companies': cumulative_companies.values
})

# Create the line plot with Plotly Express
fig = px.line(
    growth_df,
    x='Year',
    y='Cumulative Companies',
    title='Growth of Specialized Cyber Security Companies Over Time',
    markers=True
)

# Update the layout, if necessary
fig.update_layout(
    xaxis_title='Year Established',
    yaxis_title='Cumulative Number of Companies',
    showlegend=False,
    dragmode=False
)

# Display the plot
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "staticPlot": True})


# Create two columns for the pie charts
col1, col2 = st.columns(2)

with col1:
    # Total market income section
    st.markdown("""
    #### Total market income
    This bar chart displays the annual total market income for all companies, measured in Norwegian Kroner (NOK). 
    The data reveal the market's evolution. 
    """)

    # Define the years to be included in the analysis
    years = ['Income 2020', 'Income 2021', 'Income 2022']
    if include_2023_data:
        years.append('Income 2023')

    # Filter out the 'SUM' row
    df_no_sum = df[df['Organization Number'] != 'SUM']

    # Ensure the relevant columns are converted to numeric types
    for year in years:
        df_no_sum[year] = pd.to_numeric(df_no_sum[year].astype(str).str.replace(',', '').str.replace(' ', ''), errors='coerce')

    # Calculate the total incomes for the selected years
    market_growth_data = df_no_sum[years].sum()

    # Create a DataFrame suitable for plotting
    market_growth_df = pd.DataFrame({
        'Year': years,
        'Total Income': market_growth_data.values
    })

    # Create the bar chart
    fig_market_growth = px.bar(
        market_growth_df,
        x='Year',
        y='Total Income',
        title=None
    )

    # Update layout
    fig_market_growth.update_layout(
        xaxis_title="Year",
        yaxis_title="Total Market Income (NOK)",
        showlegend=False,
        autosize=True,
    )

    st.plotly_chart(fig_market_growth, use_container_width=True, config={"displayModeBar": False, "staticPlot": True})


# Yearly Percentage Change in Total Market Income - Vertical Bar Chart
with col2:
    st.markdown("""
    #### Year over Year Percentage Change in Total Market Income
    The percentage change from year to year gives a clearer picture of market trends.
    Notably, the data indicates that market growth came to a complete halt in 2023
                
    """)

    # Define the years to be included in the analysis
    years = ['Income 2020', 'Income 2021', 'Income 2022']
    if include_2023_data:
        years.append('Income 2023')

    # Extract the last row for the total incomes for the selected years
    market_growth_data = df[years].iloc[-1]
    market_growth_data = pd.to_numeric(market_growth_data, errors='coerce').fillna(0)

    # Calculating percentage change
    percentage_change = pd.Series(market_growth_data).pct_change() * 100

    # Define the year intervals for percentage change calculation
    year_intervals = ['2020-2021', '2021-2022']
    if include_2023_data:
        year_intervals.append('2022-2023')

    # Create a DataFrame for plotting
    market_growth_percentage_df = pd.DataFrame({
        'Year Interval': year_intervals,
        'Percentage Change': percentage_change[1:len(year_intervals)+1].values
    })

    # Create a vertical bar chart
    fig_market_growth_percentage = px.bar(
        market_growth_percentage_df,
        x='Year Interval',
        y='Percentage Change',
        title=None,
    )

    # Update layout to disable zoom and add percentage sign
    fig_market_growth_percentage.update_layout(
        xaxis_title="Year Interval",
        yaxis_title="Percentage Change (%)",
        yaxis=dict(
            tickformat=',.0f%',  # Format ticks as percentages
            ticksuffix='%'
        ),
        showlegend=False,
        autosize=True,
    )

    # Show the figure
    st.plotly_chart(fig_market_growth_percentage, use_container_width=True, config={"displayModeBar": False, "staticPlot": True})

# Explanation text for the sum result graph
st.markdown("""
#### Total Market Result for All Companies
This graph shows the total market result for all companies from 2020 to 2023. By summing the annual results of all companies, we can gauge the overall health of the market.
The trend indicates a substantial decline in overall market results in recent years, with significant financial losses being evident.
""")

# Ensure the columns we need are numeric
for col in ['Result 2020', 'Result 2021', 'Result 2022', 'Result 2023']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Extract the 'SUM' row for results
sum_row = df[df['Organization Number'] == 'SUM']

# Calculate the sum result for each year using the 'SUM' row and scale to MNOK
sum_result = sum_row[['Result 2020', 'Result 2021', 'Result 2022', 'Result 2023']].values[0] / 1_000_000  # Convert to millions (MNOK)

# Create a DataFrame suitable for plotting
sum_result_df = pd.DataFrame({
    'Year': ['2020', '2021', '2022', '2023'],
    'Sum Result (MNOK)': sum_result
})

# Determine the colors based on the values
colors = ['red' if val < 0 else 'blue' for val in sum_result_df['Sum Result (MNOK)']]

# Create the bar chart for sum result
fig_sum_result = px.bar(
    sum_result_df,
    x='Year',
    y='Sum Result (MNOK)',
    title='Total Market Result for All Companies',
    color=colors,
    color_discrete_map='identity'  # Ensures the colors are used as specified in the list
)

# Update layout to match the market trend graph
fig_sum_result.update_layout(
    yaxis_title='Sum Result (MNOK)',
    xaxis_title='Year',
    xaxis=dict(
        tickmode='array',
        tickvals=[2020, 2021, 2022, 2023],
        ticktext=['2020', '2021', '2022', '2023']
    ),
    hovermode='x',
    dragmode=False,
    yaxis=dict(
        tickformat=',.0f',  # Show numbers without decimal places
        tickprefix='',  # Remove 'mnok' prefix
        ticksuffix=' MNOK'  # Add ' MNOK' suffix to the tick labels
    )
)

# Display the chart
st.plotly_chart(fig_sum_result, use_container_width=True, config={"displayModeBar": False, "staticPlot": True})

st.markdown("""
#### Market Share Distribution per year
The chart(s) below clearly show that Mnemonic still holds the largest market share, followed by Netsecurity (Netsecurity and Data Equipment merged in 2023).
I calculate the market share by dividing each company's income by the total market income.
""")

# Check if 2023 market share data is enough to be displayed
if include_2023_data:
    market_share_col_2023 = 'Market share 2023'
    if market_share_col_2023 in df.columns:
        # Filter out the 'SUM' row and any empty data for the 2023 market share
        marketshare_df_2023 = df[df[market_share_col_2023].notna() & (df['Organization Number'] != 'SUM')].copy()
        marketshare_df_2023[market_share_col_2023] = pd.to_numeric(
            marketshare_df_2023[market_share_col_2023].astype(str).str.replace('%', ''), errors='coerce') / 100
        fig_2023 = px.pie(
            marketshare_df_2023,
            values=market_share_col_2023,
            names='Company Name',
            title='Market Share Distribution 2023'
        )
        st.markdown("#### Market Share Distribution 2023")
        st.plotly_chart(fig_2023, use_container_width=True, config={"displayModeBar": False, "staticPlot": True})
else:
    st.markdown("#### Market Share Distribution 2023")
    st.write("The market share distribution for 2023 will be displayed once sufficient data is available.")

with st.expander("Show Market Share Distribution 2022"):
    col1, col2 = st.columns(2)

    with col1:
        market_share_col_2022 = 'Market share 2022'
        if market_share_col_2022 in df.columns:
            marketshare_df_2022 = df[df[market_share_col_2022].notna() & (df['Organization Number'] != 'SUM')].copy()
            marketshare_df_2022[market_share_col_2022] = pd.to_numeric(
                marketshare_df_2022[market_share_col_2022].astype(str).str.replace('%', ''), errors='coerce') / 100
            fig_2022 = px.pie(
                marketshare_df_2022,
                values=market_share_col_2022,
                names='Company Name',
                title='Market Share Distribution 2022'
            )
            st.plotly_chart(fig_2022, use_container_width=True, config={"displayModeBar": False, "staticPlot": True})

with st.expander("Show Market Share Distribution 2021"):
    col1, col2 = st.columns(2)

    with col2:
        market_share_col_2021 = 'Market share 2021'
        if market_share_col_2021 in df.columns:
            # Filter out the 'SUM' row by checking if the 'Organization Number' column is not 'SUM'
            marketshare_df_2021 = df[(df[market_share_col_2021].notna()) & (df['Organization Number'] != 'SUM')].copy()
            marketshare_df_2021[market_share_col_2021] = pd.to_numeric(
                marketshare_df_2021[market_share_col_2021].astype(str).str.replace('%', ''), errors='coerce') / 100
            fig_2021 = px.pie(
                marketshare_df_2021,
                values=market_share_col_2021,
                names='Company Name',
                title='Market Share Distribution 2021'
            )

            st.plotly_chart(fig_2021, use_container_width=True, config={"displayModeBar": False, "staticPlot": True})


# Market Trend per Company (2020-2023)
st.markdown("""
#### Market Trend per Company (2020-2023)
This horizontal bar chart depicts the changes in market share for each company from 2020 to 2023.
Please note that Netsecurity and Data Equipment merged in 2023, significantly increasing Netsecurity's market share and positively impacting its market trend.
""")
trend_col = 'Market Share Trend 2020-2023'

if trend_col in df.columns:
    # Convert the column to float and handle the percentages
    df[trend_col] = (
        df[trend_col].astype(str)
        .str.replace(',', '.')
        .str.replace('−', '-')  # Standardize negative signs
        .str.rstrip('%')
        .astype(float)
    )

    # Exclude "Data Equipment AS" from the DataFrame
    trend_df = df[df['Company Name'] != 'Data Equipment AS']

    # Sort the dataframe by the trend column
    trend_df = trend_df.sort_values(by=trend_col, ascending=False)

    # Determine the colors based on the values
    colors = ['red' if val < 0 else 'blue' for val in trend_df[trend_col]]

    # Create the figure with an appropriate size
    fig_trend = px.bar(
        trend_df,
        x='Company Name',
        y=trend_col,
        title=None,
        height=600,
        color=colors,
        color_discrete_map='identity'  # Ensures the colors are used as specified in the list
    )

    # Update layout to improve visibility of small changes
    fig_trend.update_layout(
        yaxis=dict(
            type='linear',
            title='Change in Market Share (%)',
            tickformat='.0%',  # Format ticks as percentages without decimal places
            range=[min(trend_df[trend_col]), max(trend_df[trend_col])]  # Adjust the range to focus on the data
        ),
        xaxis_title="Company Name",
        hovermode='x',
        dragmode=False
    )

    # Display the chart
    st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False, "staticPlot": True})

# Result per company for 2023 in NOK
st.markdown("""
#### Result per company for 2023 in NOK
This graph shows the result for each company in NOK for 2023.
""")

# Filter the DataFrame to exclude the 'SUM' row and where 'Result 2023' is not NaN
results_df_2023 = df[df['Organization Number'] != 'SUM'][df['Result 2023'].notna()]

# Ensure the results are numeric and fill NaN with zeros if needed
results_df_2023['Result 2023'] = pd.to_numeric(results_df_2023['Result 2023'], errors='coerce').fillna(0)

# Sort the DataFrame based on 'Result 2023' for better visualization
results_df_2023.sort_values(by='Result 2023', ascending=False, inplace=True)

# Determine the colors based on the values
colors_2023 = ['red' if val < 0 else 'blue' for val in results_df_2023['Result 2023']]

# Create the figure with Plotly Express for a vertical bar chart
fig_result_2023 = px.bar(
    results_df_2023, 
    x='Company Name', 
    y='Result 2023', 
    height=600,
    title='Company Results for 2023 in NOK',
    color=colors_2023,
    color_discrete_map='identity'  # Ensures the colors are used as specified in the list
)

# Update layout to match the market trend graph
fig_result_2023.update_layout(
    yaxis_title='Result (NOK)',
    xaxis_title='Company',
    xaxis={'categoryorder':'total descending'},
    hovermode='x',
    dragmode=False
)

# Add hover data for precise values
fig_result_2023.update_traces(
    hovertemplate='%{x}: %{y:.2f} NOK<extra></extra>',
    textposition='none'
)

# Display the figure in the Streamlit app
st.plotly_chart(fig_result_2023, use_container_width=True, config={"displayModeBar": False, "staticPlot": True})

with st.expander("Show Result per company for 2022 in NOK"):
    st.markdown("""
    #### Result per company for 2022 in NOK
    This graph shows the result for each company in NOK for 2022.
    """)
    
    # Filter the DataFrame to exclude the 'SUM' row if present
    results_df = df[df['Organization Number'] != 'SUM'].copy()

    # Ensure the column values are strings before applying string operations
    results_df['Result 2022'] = results_df['Result 2022'].astype(str)

    # Ensure the results are numeric and fill NaN with zeros if needed
    results_df['Result 2022'] = pd.to_numeric(results_df['Result 2022'].str.replace(' ', '').str.replace(',', '.'), errors='coerce').fillna(0)

    # Sort the DataFrame based on 'Result 2022' for better visualization
    results_df.sort_values(by='Result 2022', ascending=False, inplace=True)

    # Determine the colors based on the values
    colors_2022 = ['red' if val < 0 else 'blue' for val in results_df['Result 2022']]

    # Create the figure with Plotly Express for a vertical bar chart
    fig_result_2022 = px.bar(
        results_df, 
        x='Company Name', 
        y='Result 2022',
        title=None, 
        height=600,
        color=colors_2022,
        color_discrete_map='identity'  # Ensures the colors are used as specified in the list
    )

    # Update layout to match the market trend graph
    fig_result_2022.update_layout(
        yaxis_title='Result (NOK)',
        xaxis_title='Company',
        xaxis={'categoryorder':'total descending'},
        hovermode='x',
        dragmode=False,
    )

    # Add hover data for precise values
    fig_result_2022.update_traces(
        hovertemplate='%{x}: %{y:.2f} NOK<extra></extra>',
        textposition='none'
    )

    st.plotly_chart(fig_result_2022, use_container_width=True, config={"displayModeBar": False, "staticPlot": True})

# Profit Margin per Company for 2023
if include_2023_data:
    st.markdown("""
    #### Profit Margin per Company for 2023
    The following bar chart illustrates the profit margin for each company in 2023.
    """)

    # Filter the DataFrame to exclude the 'SUM' row and where 'Profit margin 2023' is not NaN
    profit_margin_df_2023 = df[df['Organization Number'] != 'SUM'][df['Profit margin 2023'].notna()]

    # Convert to string and replace ',' with '.', and then convert to float
    profit_margin_df_2023['Profit margin 2023'] = (
        profit_margin_df_2023['Profit margin 2023']
        .astype(str)  # Convert to string to perform string operations
        .str.replace(',', '.')  # Replace commas with dots for decimal conversion
        .str.rstrip('%')  # Remove the percentage sign
        .astype(float)  # Convert the string to a float
    )

    # Convert percentages to decimal form by dividing by 100
    profit_margin_df_2023['Profit margin 2023'] /= 100

    # Separate the data into positive and negative profit margins
    positive_profit_margin_df = profit_margin_df_2023[profit_margin_df_2023['Profit margin 2023'] >= 0].sort_values(by='Profit margin 2023', ascending=False)
    negative_profit_margin_df = profit_margin_df_2023[profit_margin_df_2023['Profit margin 2023'] < 0].sort_values(by='Profit margin 2023', ascending=False)

    # Determine the colors based on the values
    positive_colors = ['blue' for _ in positive_profit_margin_df['Profit margin 2023']]
    negative_colors = ['red' for _ in negative_profit_margin_df['Profit margin 2023']]

    # Create the figure for positive profit margins
    fig_positive_profit_margin = px.bar(
        positive_profit_margin_df,
        x='Company Name',
        y='Profit margin 2023',
        height=600,
        color=positive_colors,
        color_discrete_map='identity'  # Ensures the colors are used as specified in the list
    )

    # Create the figure for negative profit margins
    fig_negative_profit_margin = px.bar(
        negative_profit_margin_df,
        x='Company Name',
        y='Profit margin 2023',
        height=600,
        color=negative_colors,
        color_discrete_map='identity'  # Ensures the colors are used as specified in the list
    )

    # Combine both figures into one
    fig_combined = go.Figure(data=fig_positive_profit_margin.data + fig_negative_profit_margin.data)

    # Update layout to match the market trend graph
    fig_combined.update_layout(
        yaxis_title='Profit Margin (%)',
        xaxis_title='Company',
        yaxis=dict(
            tickformat='.0%',  # Format ticks as percentages without decimal places
            range=[-1, 1]  # Extend the range to focus on the data
        ),
        hovermode='x',
        dragmode=False,
    )

    # Calculate the percentage values for the text, ensuring no decimals and multiplying by 100
    positive_profit_margin_text = (positive_profit_margin_df['Profit margin 2023'] * 100).round().astype(int).astype(str) + '%'
    negative_profit_margin_text = (negative_profit_margin_df['Profit margin 2023'] * 100).round().astype(int).astype(str) + '%'

    # Add debug prints to compare the values
    for i, (index, row) in enumerate(positive_profit_margin_df.iterrows()):
        label = positive_profit_margin_text.iloc[i]
        debug_print(f"Positive Profit Margin for {row['Company Name']}: {row['Profit margin 2023'] * 100}% (Label: {label})")
    for i, (index, row) in enumerate(negative_profit_margin_df.iterrows()):
        label = negative_profit_margin_text.iloc[i]
        debug_print(f"Negative Profit Margin for {row['Company Name']}: {row['Profit margin 2023'] * 100}% (Label: {label})")

    # Update traces to display the text within the bars, ensuring it's at the start (bottom) for positive and end (top) for negative
    fig_combined.update_traces(
        text=positive_profit_margin_text,  # Set the calculated text values
        textposition='inside',  # Position the text inside the bars
        insidetextanchor='start',  # Anchor the text to the start of the bar
        textfont=dict(
            size=12,  # Set a fixed size for the text
            color='white'  # Set the text color to white for visibility
        ),
        selector=dict(marker_color='blue')  # Apply to positive bars only
    )

    fig_combined.update_traces(
        text=negative_profit_margin_text,  # Set the calculated text values
        textposition='inside',  # Position the text inside the bars
        insidetextanchor='start',  # Anchor the text to the start of the bar
        textfont=dict(
            size=12,  # Set a fixed size for the text
            color='white'  # Set the text color to white for visibility
        ),
        selector=dict(marker_color='red')  # Apply to negative bars only
    )

    # Display the updated figure in the Streamlit app
    st.plotly_chart(fig_combined, use_container_width=True, config={"displayModeBar": False, "staticPlot": True})

else:
    # Display a message if not enough data is available for 2023
    st.markdown("""
    #### Profit Margin per Company for 2023
    The graph illustrating the profit margin for each company in 2023 will be displayed once sufficient data is available.
    """)

with st.expander("Show Profit Margin per Company for 2022"):
    st.markdown("""
    #### Profit Margin per Company for 2022
    The following bar chart illustrates the profit margin for each company in 2022.
    """)
    
    # Define the profit margin column for 2022
    profit_margin_col = 'Profit margin 2022'

    # Convert to string and replace ',' with '.', and then convert to float
    df[profit_margin_col] = (
        df[profit_margin_col]
        .astype(str)  # Convert to string to perform string operations
        .str.replace(',', '.')  # Replace commas with dots for decimal conversion
        .str.rstrip('%')  # Remove the percentage sign
        .astype(float)  # Convert the string to a float
    )

    # Convert percentages to decimal form by dividing by 100
    df[profit_margin_col] = df[profit_margin_col] / 100

    # Separate the data into positive and negative profit margins
    positive_profit_margin_df = df[df[profit_margin_col] >= 0].sort_values(by=profit_margin_col, ascending=False)
    negative_profit_margin_df = df[df[profit_margin_col] < 0].sort_values(by=profit_margin_col, ascending=False)

    # Determine the colors based on the values
    positive_colors = ['blue' for _ in positive_profit_margin_df[profit_margin_col]]
    negative_colors = ['red' for _ in negative_profit_margin_df[profit_margin_col]]

    # Create the figure for positive profit margins
    fig_positive_profit_margin = px.bar(
        positive_profit_margin_df,
        x='Company Name',
        y=profit_margin_col,
        height=600,
        color=positive_colors,
        color_discrete_map='identity'  # Ensures the colors are used as specified in the list
    )

    # Create the figure for negative profit margins
    fig_negative_profit_margin = px.bar(
        negative_profit_margin_df,
        x='Company Name',
        y=profit_margin_col,
        height=600,
        color=negative_colors,
        color_discrete_map='identity'  # Ensures the colors are used as specified in the list
    )

    # Combine both figures into one
    fig_combined = go.Figure(data=fig_positive_profit_margin.data + fig_negative_profit_margin.data)

    # Update layout to match the market trend graph
    fig_combined.update_layout(
        yaxis_title='Profit Margin (%)',
        xaxis_title='Company',
        yaxis=dict(
            tickformat='.0%',  # Format ticks as percentages without decimal places
            range=[-1, 1]  # Extend the range to focus on the data
        ),
        hovermode='x',
        dragmode=False,
    )

    # Calculate the percentage values for the text, ensuring no decimals and multiplying by 100
    positive_profit_margin_text = (positive_profit_margin_df[profit_margin_col] * 100).round().astype(int).astype(str) + '%'
    negative_profit_margin_text = (negative_profit_margin_df[profit_margin_col] * 100).round().astype(int).astype(str) + '%'

    # Add debug prints to compare the values
    for i, (index, row) in enumerate(positive_profit_margin_df.iterrows()):
        label = positive_profit_margin_text.iloc[i]
        debug_print(f"Positive Profit Margin for {row['Company Name']}: {row[profit_margin_col] * 100}% (Label: {label})")
    for i, (index, row) in enumerate(negative_profit_margin_df.iterrows()):
        label = negative_profit_margin_text.iloc[i]
        debug_print(f"Negative Profit Margin for {row['Company Name']}: {row[profit_margin_col] * 100}% (Label: {label})")

    # Update traces to display the text within the bars, ensuring it's at the start (bottom) for positive and end (top) for negative
    fig_combined.update_traces(
        text=positive_profit_margin_text,  # Set the calculated text values
        textposition='inside',  # Position the text inside the bars
        insidetextanchor='start',  # Anchor the text to the start of the bar
        textfont=dict(
            size=12,  # Set a fixed size for the text
            color='white'  # Set the text color to white for visibility
        ),
        selector=dict(marker_color='blue')  # Apply to positive bars only
    )

    fig_combined.update_traces(
        text=negative_profit_margin_text,  # Set the calculated text values
        textposition='inside',  # Position the text inside the bars
        insidetextanchor='start',  # Anchor the text to the start of the bar
        textfont=dict(
            size=12,  # Set a fixed size for the text
            color='white'  # Set the text color to white for visibility
        ),
        selector=dict(marker_color='red')  # Apply to negative bars only
    )

    # Display the updated figure in the Streamlit app
    st.plotly_chart(fig_combined, use_container_width=True, config={"displayModeBar": False, "staticPlot": True})

if 'last_updated_time' not in st.session_state:
    get_and_update_last_updated_time()

# Format the datetime object to exclude seconds when converting to string
last_updated_time_formatted = st.session_state['last_updated_time'].strftime('%Y-%m-%d %H:%M UTC')

# Display the button and the last check time
st.subheader("Check if new data is available")
st.write(f"Note: The check for new data button can only be clicked if it's been more than 20 minutes since the last check. Last check: {last_updated_time_formatted}")
button_disabled = is_button_disabled()  # Check if the button should be disabled

# When the button is clicked
if st.button("Check for new data", disabled=button_disabled):
    st.session_state['last_updated_time'] = datetime.now()
    # Fetch the existing dataframe when the button is clicked
    existing_df = fetch_existing_df()
    
    selskapsdata_worksheet = sh.worksheet('title', 'selskapsdata')
    other_worksheet = sh.worksheet('title', 'other')

    if existing_df is not None:
        # Convert organization number to string and strip whitespace
        existing_df['Organization Number'] = existing_df['Organization Number'].astype(str).str.strip()

        # Create a dictionary to map org numbers to company names for fast lookup
        org_number_to_company = pd.Series(existing_df['Company Name'].values, index=existing_df['Organization Number']).to_dict()

        total_org_numbers = len(existing_df['Organization Number'].tolist())

        # Initialize progress bar and status text
        progress_bar = st.progress(0)
        status_text = st.text("Checking if new data is available...")

        detailed_update_summary = []  # List to store detailed updates

        # Iterate over each organization number and check for new financial data
        for index, org_number in enumerate(existing_df['Organization Number']):
            # Attempt to retrieve the company name from the mapping
            company_name = org_number_to_company.get(org_number, "Unknown Company")

            # Update progress bar and status text
            progress_bar.progress((index + 1) / total_org_numbers)
            status_text.text(f"Checking data for {company_name} ({org_number})")

            # Skip non-numeric org numbers and "SUM"
            if not org_number.isdigit() or org_number.upper() == "SUM":
                continue

            # Fetch new financial data for the organization number
            financial_data = get_financial_data(org_number)
            if financial_data:
                updates = update_google_sheet(selskapsdata_worksheet, org_number, financial_data, existing_df)
                if updates:
                    for update in updates:
                        update['company_name'] = company_name  # Add company name to each update
                    detailed_update_summary.extend(updates)  # Add the updates to the summary

        # Clear progress bar and status text once checking is done
        progress_bar.empty()
        status_text.empty()


        # Display update summary
        if detailed_update_summary:
            st.write("### The following updates were made:")

            # Create a DataFrame to display the updates in a table format
            updates_df = pd.DataFrame(detailed_update_summary)
            updates_df['Income Change'] = updates_df.apply(lambda row: f"{row['income']['old']} → {row['income']['new']}", axis=1)
            updates_df['Result Change'] = updates_df.apply(lambda row: f"{row['result']['old']} → {row['result']['new']}", axis=1)

            # Format the numbers with commas and no decimal places
            updates_df['Income Change'] = updates_df['Income Change'].str.replace(r'(\d+)', lambda x: "{:,}".format(int(x.group())), regex=True)
            updates_df['Result Change'] = updates_df['Result Change'].str.replace(r'(\d+)', lambda x: "{:,}".format(int(x.group())), regex=True)

            # Drop the old income/result columns as they are now combined into 'Change' columns
            updates_df.drop(columns=['income', 'result'], inplace=True)

            # Rename the columns for better readability
            updates_df.rename(columns={'org_number': 'Org Number', 'accounting_year': 'Year'}, inplace=True)

            # Remove the index before displaying
            updates_df.reset_index(drop=True, inplace=True)

            # Convert DataFrame to HTML and remove the index
            updates_df_html = updates_df.to_html(index=False)

            # Use markdown to display the table without the index column
            st.markdown(updates_df_html, unsafe_allow_html=True)
        else:
            st.success("No changes were necessary. All data is up to date.")

        # Update the 'Last updated' time in the Google Sheet after checking all companies
        update_last_update_time_in_sheet('other', 'B1', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

# Overview Section
st.markdown("---")
st.markdown("""
### Company Overview

This section provides a detailed overview of each company's type, specialization, and services offered. Use the table to easily find a cyber security company that suits your needs.

- **Service or Software Provider**: Service Provider or Software Provider.
- **Full Service / Specialization / Generalist**:
  - **Full Service**: Typically larger companies offering comprehensive services (one stop shop).
  - **Specialized**: Focuses on a single domain (boutique/experts)
  - **Generalist**: Covers multiple areas but not as extensive as full service company.

If you believe there is an error in the classification of your company, please contact me.
""")

# Select relevant columns for the overview and rename them
overview_columns = ['Company Name', 'Established', 'Type', 'Specialization', 'Website URL']
rename_columns = {
    'Type': 'Service or Software Provider',
    'Company type': 'Service Model',
    'Specialization': 'Specialization / Generalist / Full Service'
}

# Check for missing columns
missing_columns = [col for col in overview_columns if col not in df.columns]
debug_print(f"Overview Columns: {overview_columns}")
debug_print(f"Missing Columns: {missing_columns}")

if missing_columns:
    st.error(f"One or more overview columns are missing from the dataset: {missing_columns}")
else:
    # Select the overview columns and rename them
    df_overview = df[overview_columns].dropna(subset=['Company Name']).copy()
    df_overview.rename(columns=rename_columns, inplace=True)
    debug_print(f"DataFrame Overview Shape: {df_overview.shape}")
    debug_print(f"DataFrame Overview Columns: {df_overview.columns}")

    # Reset index to start from 1
    df_overview.index = df_overview.index + 1

    # Column configurations for link columns
    column_config = {
        "Website URL": st.column_config.LinkColumn("Website URL")
    }

    # Display the DataFrame using st.data_editor with link column configurations and set use_container_width to True
    st.data_editor(data=df_overview, column_config=column_config, height=1400, use_container_width=True)
    debug_print("Data Editor Displayed with Link Columns")
