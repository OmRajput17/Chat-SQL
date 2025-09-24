import streamlit as st
from pathlib import Path
from langchain_community.utilities import SQLDatabase
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
import pandas as pd
import re
import json
from datetime import datetime

import os
from dotenv import load_dotenv
load_dotenv()

### Setup LANGCHAIN TRACING
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")

### Setting up streamlit app
st.set_page_config(page_title="CRUD Chat SQL", page_icon="🦜", layout="wide")
st.title("🦜 CRUD Chat SQL : Full Database Operations")

LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

### Sidebar Configuration
st.sidebar.header("Database Configuration")
radio_opt = ["Use SQLLite3 Database - student.db", "Connect to your own MySQL database"]
selected_opt = st.sidebar.radio(label="Choose the DB", options=radio_opt)

# CRUD Operations Toggle
st.sidebar.header("Operation Settings")
enable_crud = st.sidebar.checkbox("Enable CRUD Operations", value=False, help="⚠️ This allows INSERT, UPDATE, DELETE operations")
require_confirmation = st.sidebar.checkbox("Require Confirmation for Write Operations", value=True)
max_affected_rows = st.sidebar.number_input("Max rows for UPDATE/DELETE", min_value=1, max_value=1000, value=100)

# Display Options
st.sidebar.header("Display Settings")
show_table_format = st.sidebar.checkbox("Show results in table format", value=True, help="Display query results as interactive tables")
max_display_rows = st.sidebar.number_input("Max rows to display", min_value=5, max_value=1000, value=100)

if enable_crud:
    st.sidebar.warning("⚠️ CRUD operations enabled! Use with caution.")

if radio_opt.index(selected_opt) == 1:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("Provide MySQL Host Name")
    mysql_user = st.sidebar.text_input("MYSQL User")
    mysql_password = st.sidebar.text_input("MYSQL Password", type="password")
    mysql_db = st.sidebar.text_input("MYSQL DATABASE")
else:
    db_uri = LOCALDB

if not db_uri:
    st.info("Please enter the DB info and uri")

if not groq_api_key:
    st.info("Please enter a valid API key")

# Initialize LLM
llm = ChatGroq(
    groq_api_key=groq_api_key, 
    model_name="llama-3.1-8b-instant", 
    streaming=True,
    temperature=0
)

#### Connecting to the DATABASE
@st.cache_resource(ttl="2h")
def configure_db(db_uri, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None, enable_crud=False):
    if db_uri == LOCALDB:
        dbFile_path = (Path(__file__).parent/"student.db").absolute()
        print(dbFile_path)
        if enable_crud:
            # Read-write connection for CRUD operations
            creator = lambda: sqlite3.connect(f"file:{dbFile_path}", uri=True)
        else:
            # Read-only connection for safety
            creator = lambda: sqlite3.connect(f"file:{dbFile_path}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    elif db_uri == MYSQL:
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.info("Please provide all MYSQL Connection Details")
            st.stop()
        return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))

# Configure database with CRUD settings
if db_uri == MYSQL:
    db = configure_db(
        db_uri=db_uri,
        mysql_host=mysql_host,
        mysql_user=mysql_user,
        mysql_password=mysql_password,
        mysql_db=mysql_db,
        enable_crud=enable_crud
    )
else:
    db = configure_db(db_uri=db_uri, enable_crud=enable_crud)

### Enhanced SQL Agent Class with CRUD Support and Tabular Output
class EnhancedSQLAgent:
    def __init__(self, db, llm, enable_crud=False, require_confirmation=True, max_affected_rows=100):
        self.db = db
        self.llm = llm
        self.schema_info = None
        self.enable_crud = enable_crud
        self.require_confirmation = require_confirmation
        self.max_affected_rows = max_affected_rows
        
    def get_schema_info(self):
        """Get database schema information"""
        try:
            tables = self.db.get_usable_table_names()
            schema_info = {}
            for table in tables:
                try:
                    schema = self.db.get_table_info([table])
                    schema_info[table] = schema
                except Exception as e:
                    schema_info[table] = f"Error getting schema: {str(e)}"
            return tables, schema_info
        except Exception as e:
            return [], {"error": f"Error getting schema: {str(e)}"}
    
    def analyze_query_type(self, sql_query):
        """Analyze the type of SQL query"""
        sql_upper = sql_query.upper().strip()
        
        if sql_upper.startswith('SELECT'):
            return 'READ'
        elif sql_upper.startswith('INSERT'):
            return 'CREATE'
        elif sql_upper.startswith('UPDATE'):
            return 'UPDATE'
        elif sql_upper.startswith('DELETE'):
            return 'DELETE'
        elif any(sql_upper.startswith(cmd) for cmd in ['DROP', 'ALTER', 'CREATE', 'TRUNCATE']):
            return 'DDL'
        else:
            return 'UNKNOWN'
    
    def validate_sql_query(self, sql_query):
        """Validate SQL query for security and safety"""
        dangerous_patterns = [
            r';\s*(DROP|ALTER|TRUNCATE)',  # Multiple statements with dangerous operations
            r'--.*?(DROP|DELETE|UPDATE)',  # Comments hiding dangerous operations
            r'/\*.*?(DROP|DELETE|UPDATE).*?\*/',  # Multi-line comments
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, sql_query, re.IGNORECASE | re.DOTALL):
                return False, "Query contains potentially dangerous patterns"
        
        # Check for multiple statements
        if ';' in sql_query.strip() and not sql_query.strip().endswith(';'):
            return False, "Multiple SQL statements not allowed"
        
        return True, "Query is valid"
    
    def estimate_affected_rows(self, sql_query, query_type):
        """Estimate number of rows that will be affected by UPDATE/DELETE"""
        if query_type not in ['UPDATE', 'DELETE']:
            return 0
        
        try:
            # Convert UPDATE/DELETE to SELECT COUNT to estimate affected rows
            if query_type == 'UPDATE':
                # Extract table and WHERE clause from UPDATE
                match = re.search(r'UPDATE\s+(\w+)\s+SET.*?(WHERE.*?)(?:$|;)', sql_query, re.IGNORECASE | re.DOTALL)
                if match:
                    table, where_clause = match.groups()
                    count_query = f"SELECT COUNT(*) FROM {table} {where_clause}"
                else:
                    # No WHERE clause - affects all rows
                    table_match = re.search(r'UPDATE\s+(\w+)', sql_query, re.IGNORECASE)
                    if table_match:
                        table = table_match.group(1)
                        count_query = f"SELECT COUNT(*) FROM {table}"
                    else:
                        return -1
            
            elif query_type == 'DELETE':
                # Extract table and WHERE clause from DELETE
                match = re.search(r'DELETE\s+FROM\s+(\w+)(?:\s+(WHERE.*?))?(?:$|;)', sql_query, re.IGNORECASE | re.DOTALL)
                if match:
                    table, where_clause = match.groups()
                    if where_clause:
                        count_query = f"SELECT COUNT(*) FROM {table} {where_clause}"
                    else:
                        count_query = f"SELECT COUNT(*) FROM {table}"
                else:
                    return -1
            
            # Execute count query
            result = self.db.run(count_query)
            # Parse the result to get the count
            try:
                count = int(result.split('\n')[0] if '\n' in result else result)
                return count
            except (ValueError, IndexError):
                return -1
                
        except Exception as e:
            return -1
    
    def parse_sql_result_to_dataframe(self, result_string, sql_query):
        """
        Parse SQL result string to pandas DataFrame for tabular display
        """
        try:
            if not result_string or result_string.strip() == "":
                return None, "No results returned"
            
            # Split the result into lines
            lines = result_string.strip().split('\n')
            
            if len(lines) <= 1:
                # If only one line, it might be a single value or error
                return None, result_string
            
            # Try to detect if it's tabular data
            # Look for consistent delimiters (pipes, tabs, commas)
            if '|' in lines[0]:
                # Pipe-separated format
                headers = [col.strip() for col in lines[0].split('|') if col.strip()]
                data_rows = []
                
                for line in lines[1:]:
                    if line.strip() and '|' in line:
                        row = [col.strip() for col in line.split('|') if col.strip()]
                        if len(row) == len(headers):
                            data_rows.append(row)
                
                if data_rows:
                    df = pd.DataFrame(data_rows, columns=headers)
                    return df, None
            
            # Try comma-separated or tab-separated
            elif '\t' in lines[0] or ',' in lines[0]:
                delimiter = '\t' if '\t' in lines[0] else ','
                headers = [col.strip() for col in lines[0].split(delimiter)]
                data_rows = []
                
                for line in lines[1:]:
                    if line.strip():
                        row = [col.strip() for col in line.split(delimiter)]
                        if len(row) == len(headers):
                            data_rows.append(row)
                
                if data_rows:
                    df = pd.DataFrame(data_rows, columns=headers)
                    return df, None
            
            # If no clear tabular structure, try to parse as simple rows
            # This handles cases where each line is a record
            if len(lines) > 1:
                # Try to create a simple single-column DataFrame
                df = pd.DataFrame(lines, columns=['Result'])
                return df, None
            
            return None, result_string
            
        except Exception as e:
            return None, f"Error parsing results: {str(e)}\nRaw result: {result_string}"
    
    def execute_sql_query_direct(self, sql_query):
        """Execute SQL query and return raw results"""
        try:
            # For SELECT queries, try to get results as DataFrame directly
            if sql_query.strip().upper().startswith('SELECT'):
                # Get the SQLAlchemy engine from the database
                engine = self.db._engine
                df = pd.read_sql_query(sql_query, engine)
                return df, None
            else:
                # For non-SELECT queries, use the standard method
                result = self.db.run(sql_query)
                return None, result
        except Exception as e:
            try:
                # Fallback to standard method
                result = self.db.run(sql_query)
                return None, result
            except Exception as fallback_e:
                return None, f"Error executing query: {str(fallback_e)}"
    
    def generate_sql_query(self, question, tables, schema_info):
        """Generate SQL query based on question and schema"""
        schema_context = "Database Schema:\n"
        for table, schema in schema_info.items():
            schema_context += f"\nTable: {table}\n{schema}\n"
        
        # Different system prompts based on CRUD settings
        if self.enable_crud:
            operation_rules = """Rules:
1. You can use SELECT, INSERT, UPDATE, DELETE operations
2. For SELECT: Limit results to 50 unless specified otherwise
3. For INSERT: Ensure all required fields are provided
4. For UPDATE/DELETE: Always include WHERE clause unless explicitly asked to update/delete all records
5. Use proper column names as shown in the schema
6. Order results by relevant columns when possible
7. Be careful with UPDATE and DELETE operations
8. Return only the SQL query"""
        else:
            operation_rules = """Rules:
1. Only use SELECT statements (no INSERT, UPDATE, DELETE)
2. Limit results to 50 unless specified otherwise
3. Use proper column names as shown in the schema
4. Order results by relevant columns when possible
5. Return only the SQL query"""

        system_prompt = f"""You are a SQL expert. Given a database schema and a question, generate a syntactically correct SQL query.

{schema_context}

{operation_rules}

Question: {question}

SQL Query:"""

        try:
            messages = [SystemMessage(content=system_prompt)]
            response = self.llm.invoke(messages)
            
            sql_query = response.content.strip()
            
            # Clean up the query
            if sql_query.startswith("```sql"):
                sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
            elif sql_query.startswith("```"):
                sql_query = sql_query.replace("```", "").strip()
                
            return sql_query
        except Exception as e:
            return f"Error generating query: {str(e)}"
    
    def execute_query_with_confirmation(self, sql_query, query_type):
        """Execute query with confirmation for write operations"""
        try:
            if query_type == 'READ':
                # For SELECT queries, try to get DataFrame directly
                df, result_text = self.execute_sql_query_direct(sql_query)
                if df is not None:
                    return True, {"type": "dataframe", "data": df}
                else:
                    return True, {"type": "text", "data": result_text}
            
            elif query_type in ['CREATE', 'UPDATE', 'DELETE'] and self.enable_crud:
                if query_type in ['UPDATE', 'DELETE']:
                    # Estimate affected rows
                    affected_rows = self.estimate_affected_rows(sql_query, query_type)
                    
                    if affected_rows > self.max_affected_rows:
                        return False, f"Operation would affect {affected_rows} rows, which exceeds the limit of {self.max_affected_rows} rows. Please refine your query."
                    
                    if self.require_confirmation:
                        return "CONFIRMATION_REQUIRED", {
                            "query": sql_query,
                            "type": query_type,
                            "estimated_rows": affected_rows
                        }
                
                # Execute the query
                result = self.db.run(sql_query)
                return True, {"type": "text", "data": f"✅ {query_type} operation completed successfully.\n\nResult: {result}"}
            
            else:
                return False, f"{query_type} operations are not enabled. Enable CRUD operations in the sidebar to use this functionality."
                
        except Exception as e:
            return False, f"Error executing query: {str(e)}"
    
    def process_question(self, question):
        """Main method to process user question"""
        try:
            # Step 1: Get schema information
            tables, schema_info = self.get_schema_info()
            
            if not tables:
                return "Error: Could not retrieve database schema information."
            
            # Step 2: Generate SQL query
            sql_query = self.generate_sql_query(question, tables, schema_info)
            
            if sql_query.startswith("Error"):
                return sql_query
            
            # Step 3: Validate query
            is_valid, validation_msg = self.validate_sql_query(sql_query)
            if not is_valid:
                return f"❌ Invalid query: {validation_msg}"
            
            # Step 4: Analyze query type
            query_type = self.analyze_query_type(sql_query)
            
            # Step 5: Execute query
            success, result = self.execute_query_with_confirmation(sql_query, query_type)
            
            if success == "CONFIRMATION_REQUIRED":
                return "CONFIRMATION_REQUIRED", result
            elif success:
                return {
                    "sql_query": sql_query,
                    "query_type": query_type,
                    "result": result
                }
            else:
                return f"❌ **Error:** {result}\n\n**Generated Query:**\n```sql\n{sql_query}\n```"
            
        except Exception as e:
            return f"Error processing question: {str(e)}"

# Initialize enhanced agent
enhanced_agent = EnhancedSQLAgent(
    db, llm, 
    enable_crud=enable_crud, 
    require_confirmation=require_confirmation,
    max_affected_rows=max_affected_rows
)

### Session State Management
if "messages" not in st.session_state or st.sidebar.button("Clear Message History"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you with your database? I can perform read operations, and if enabled, CREATE, UPDATE, and DELETE operations."}]

if "pending_confirmation" not in st.session_state:
    st.session_state["pending_confirmation"] = None

# Display chat history
for msg in st.session_state.messages:
    if isinstance(msg["content"], dict):
        # Handle structured responses with tables
        st.chat_message(msg["role"]).write("**Query executed successfully!**")
    else:
        st.chat_message(msg["role"]).write(msg["content"])

# Handle pending confirmations
if st.session_state.pending_confirmation:
    st.warning("⚠️ **Confirmation Required**")
    confirmation_data = st.session_state.pending_confirmation
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **Operation:** {confirmation_data['type']}
        **Estimated affected rows:** {confirmation_data['estimated_rows']}
        **Query:** 
        ```sql
        {confirmation_data['query']}
        ```
        """)
    
    with col2:
        if st.button("✅ Confirm & Execute", type="primary"):
            try:
                result = enhanced_agent.db.run(confirmation_data['query'])
                response = f"✅ {confirmation_data['type']} operation completed successfully.\n\nResult: {result}"
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.pending_confirmation = None
                st.rerun()
            except Exception as e:
                error_msg = f"❌ Error executing confirmed query: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.session_state.pending_confirmation = None
                st.rerun()
        
        if st.button("❌ Cancel"):
            st.session_state.messages.append({"role": "assistant", "content": "Operation cancelled by user."})
            st.session_state.pending_confirmation = None
            st.rerun()

### User Query Input
user_query = st.chat_input(placeholder="Ask anything about your database (e.g., 'Show all students', 'Add a new student named John', 'Update student grades')")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Processing your query..."):
                result = enhanced_agent.process_question(user_query)
            
            if isinstance(result, tuple) and result[0] == "CONFIRMATION_REQUIRED":
                st.session_state.pending_confirmation = result[1]
                st.rerun()
            elif isinstance(result, dict) and "sql_query" in result:
                # Handle structured response with potential DataFrame
                st.write(f"**Generated SQL Query ({result['query_type']}):**")
                st.code(result['sql_query'], language='sql')
                
                if result['result']['type'] == 'dataframe':
                    df = result['result']['data']
                    
                    # Display row count
                    st.write(f"**Results ({len(df)} rows):**")
                    
                    if show_table_format:
                        # Display as interactive table
                        if len(df) > max_display_rows:
                            st.warning(f"Showing first {max_display_rows} rows out of {len(df)} total rows")
                            st.dataframe(df.head(max_display_rows), use_container_width=True)
                        else:
                            st.dataframe(df, use_container_width=True)
                        
                        # Add download button for the data
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download as CSV",
                            data=csv,
                            file_name="query_results.csv",
                            mime="text/csv"
                        )
                    else:
                        # Display as raw text
                        st.text(df.to_string())
                        
                elif result['result']['type'] == 'text':
                    st.write("**Results:**")
                    st.text(result['result']['data'])
                
                # Store the structured result for history
                display_content = {
                    "query": result['sql_query'],
                    "type": result['query_type'],
                    "row_count": len(result['result']['data']) if result['result']['type'] == 'dataframe' else None
                }
                st.session_state.messages.append({"role": "assistant", "content": display_content})
                
            else:
                # Handle string responses (errors, etc.)
                st.session_state.messages.append({"role": "assistant", "content": result})
                st.write(result)
                
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Display operation statistics in sidebar
if enable_crud:
    st.sidebar.header("Operation Statistics")
    st.sidebar.info(f"""
    **Current Session:**
    - CRUD Operations: Enabled ✅
    - Confirmation Required: {require_confirmation}
    - Max Affected Rows: {max_affected_rows}
    - Table Display: {show_table_format}
    - Max Display Rows: {max_display_rows}
    """)

# Help section
with st.sidebar.expander("💡 Example Queries"):
    st.write("""
    **READ Operations:**
    - "Show all students"
    - "Find students with grade > 85"
    - "Count total students"
    - "Show top 10 students by grade"
    
    **CREATE Operations:**
    - "Add a new student named John with age 20"
    - "Insert a record for student ID 123"
    
    **UPDATE Operations:**
    - "Update John's grade to 95"
    - "Change all grades below 60 to 60"
    
    **DELETE Operations:**
    - "Delete student with ID 123"
    - "Remove all students with grade < 50"
    """)