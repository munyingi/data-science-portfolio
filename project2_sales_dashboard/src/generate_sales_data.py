"""
Generate realistic Superstore sales dataset
Author: Samwel Munyingi
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration
num_records = 10000
start_date = datetime(2020, 1, 1)
end_date = datetime(2023, 12, 31)

# Define categories and subcategories
categories = {
    'Furniture': ['Bookcases', 'Chairs', 'Tables', 'Furnishings'],
    'Office Supplies': ['Storage', 'Art', 'Binders', 'Paper', 'Appliances', 'Labels', 'Envelopes', 'Fasteners', 'Supplies'],
    'Technology': ['Phones', 'Accessories', 'Copiers', 'Machines']
}

# Regions and states
regions = {
    'East': ['New York', 'Pennsylvania', 'Massachusetts', 'Connecticut', 'New Jersey'],
    'West': ['California', 'Washington', 'Oregon', 'Nevada', 'Arizona'],
    'Central': ['Texas', 'Illinois', 'Ohio', 'Michigan', 'Indiana'],
    'South': ['Florida', 'Georgia', 'North Carolina', 'Virginia', 'Tennessee']
}

# Customer segments
segments = ['Consumer', 'Corporate', 'Home Office']

# Ship modes
ship_modes = ['Standard Class', 'Second Class', 'First Class', 'Same Day']

# Generate data
data = []

for i in range(num_records):
    # Generate dates
    order_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
    ship_date = order_date + timedelta(days=random.randint(1, 7))
    
    # Select region and state
    region = random.choice(list(regions.keys()))
    state = random.choice(regions[region])
    city = f"{state[:3].upper()}-City-{random.randint(1, 5)}"
    
    # Select category and subcategory
    category = random.choice(list(categories.keys()))
    sub_category = random.choice(categories[category])
    
    # Generate product name
    product_id = f"{category[:3].upper()}-{sub_category[:3].upper()}-{random.randint(1000, 9999)}"
    product_name = f"{sub_category} {random.choice(['Pro', 'Elite', 'Standard', 'Premium', 'Basic'])} {random.randint(100, 999)}"
    
    # Customer info
    customer_id = f"CUS-{random.randint(10000, 99999)}"
    customer_name = f"Customer {random.randint(1, 1000)}"
    segment = random.choice(segments)
    
    # Order info
    order_id = f"ORD-{order_date.year}-{random.randint(10000, 99999)}"
    ship_mode = random.choice(ship_modes)
    
    # Sales metrics
    quantity = random.randint(1, 15)
    
    # Price varies by category
    if category == 'Technology':
        unit_price = random.uniform(50, 2000)
    elif category == 'Furniture':
        unit_price = random.uniform(30, 1500)
    else:  # Office Supplies
        unit_price = random.uniform(5, 500)
    
    sales = quantity * unit_price
    
    # Discount (0-40%)
    discount = random.choice([0, 0, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4])
    
    # Profit margin varies by category and discount
    if category == 'Technology':
        base_margin = random.uniform(0.15, 0.35)
    elif category == 'Furniture':
        base_margin = random.uniform(0.10, 0.25)
    else:
        base_margin = random.uniform(0.20, 0.40)
    
    # Discount reduces margin
    actual_margin = base_margin - (discount * 0.5)
    profit = sales * actual_margin
    
    # Postal code
    postal_code = random.randint(10000, 99999)
    
    # Create record
    record = {
        'Order ID': order_id,
        'Order Date': order_date.strftime('%Y-%m-%d'),
        'Ship Date': ship_date.strftime('%Y-%m-%d'),
        'Ship Mode': ship_mode,
        'Customer ID': customer_id,
        'Customer Name': customer_name,
        'Segment': segment,
        'Country': 'United States',
        'City': city,
        'State': state,
        'Postal Code': postal_code,
        'Region': region,
        'Product ID': product_id,
        'Category': category,
        'Sub-Category': sub_category,
        'Product Name': product_name,
        'Sales': round(sales, 2),
        'Quantity': quantity,
        'Discount': discount,
        'Profit': round(profit, 2)
    }
    
    data.append(record)

# Create DataFrame
df = pd.DataFrame(data)

# Sort by order date
df = df.sort_values('Order Date').reset_index(drop=True)

# Add Row ID
df.insert(0, 'Row ID', range(1, len(df) + 1))

# Save to CSV
output_path = '../data/Sample-Superstore.csv'
df.to_csv(output_path, index=False)

print(f"Dataset created successfully!")
print(f"Total records: {len(df)}")
print(f"Date range: {df['Order Date'].min()} to {df['Order Date'].max()}")
print(f"Total Sales: ${df['Sales'].sum():,.2f}")
print(f"Total Profit: ${df['Profit'].sum():,.2f}")
print(f"Saved to: {output_path}")
