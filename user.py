import pandas as pd
import os
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

class User(UserMixin):
    """User model for authentication"""
    
    def __init__(self, id, username, password_hash):
        self.id = str(id)
        self.username = username
        self.password_hash = password_hash
    
    def check_password(self, password):
        """Check if the provided password matches the hash"""
        try:
            return check_password_hash(self.password_hash, password)
        except Exception as e:
            print(f"Error checking password: {e}")
            return False
    
    @staticmethod
    def get(user_id):
        """Get user by ID"""
        try:
            # Ensure file exists
            if not os.path.exists('data/users.csv'):
                print("Users file not found")
                return None
                
            users_df = pd.read_csv('data/users.csv')
            user_id = int(user_id)
            
            if 0 <= user_id < len(users_df):
                user_data = users_df.iloc[user_id]
                return User(
                    id=user_id,
                    username=user_data['username'],
                    password_hash=user_data['password_hash']
                )
        except Exception as e:
            print(f"Error getting user: {e}")
        return None
    
    @staticmethod
    def find_by_username(username):
        """Find user by username"""
        try:
            # Ensure file exists
            if not os.path.exists('data/users.csv'):
                print("Users file not found")
                return None
                
            users_df = pd.read_csv('data/users.csv')
            user_rows = users_df[users_df['username'] == username]
            
            if not user_rows.empty:
                idx = user_rows.index[0]
                return User(
                    id=idx,
                    username=user_rows.iloc[0]['username'],
                    password_hash=user_rows.iloc[0]['password_hash']
                )
        except Exception as e:
            print(f"Error finding user: {e}")
        return None
    
    @staticmethod
    def create(username, password):
        """Create new user with hashed password"""
        try:
            # Ensure data directory exists
            os.makedirs('data', exist_ok=True)
            
            # Create file if it doesn't exist
            if not os.path.exists('data/users.csv'):
                users_df = pd.DataFrame(columns=["username", "password_hash"])
                users_df.to_csv('data/users.csv', index=False)
                print("Created new users.csv file")
            
            # Read existing users
            users_df = pd.read_csv('data/users.csv')
            
            # Check if username exists
            if username in users_df['username'].values:
                print(f"Username {username} already exists")
                return None
            
            # Hash the password
            password_hash = generate_password_hash(password)
            
            # Add new user
            new_user = pd.DataFrame([[username, password_hash]], 
                                   columns=["username", "password_hash"])
            users_df = pd.concat([users_df, new_user], ignore_index=True)
            users_df.to_csv('data/users.csv', index=False)
            
            print(f"User {username} created successfully")
            
            # Return new user
            return User(
                id=len(users_df) - 1,
                username=username,
                password_hash=password_hash
            )
        except Exception as e:
            print(f"Error creating user: {e}")
            return None