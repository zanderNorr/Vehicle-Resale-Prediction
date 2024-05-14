import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# opens the uncleaned CSV and starts cleaning the data
# valid year is set from 2000 - 2015
# valid odometer (milage) is set from 5 to 300000
# valid selling rpice is set from 2000 - 200000
def step_one():
    df = pd.read_csv('car_prices.csv')
    valid_dates = (df['year'] >= 2000) & (df['year'] <= 2015) 
    valid_miles = (df['odometer'] >= 5) & (df['odometer'] <= 300000)
    valid_prices = (df['sellingprice'] >= 2000) & (df['sellingprice'] <= 200000)
    valid_data = valid_dates & valid_miles & valid_prices
    df = df[valid_data]
    df.dropna(inplace=True)
    return df

# further data preprocessing
# all incorrectly named styles are correctly renamed
# all vehicles based on their brandage are given a new category to detail their level of luxury
def step_two(dataframe):

    df = dataframe    

    car_brands = [
        "Audi", "BMW", "Cadillac", "Jaguar", "Land Rover", "Lexus", "Lincoln", "Mercedes-Benz", "Volvo", 
        "Acura", "Infiniti", "Porsche", "Tesla", "Maserati", "Bentley", "Rolls-Royce", "Aston Martin",
        "Kia", "Nissan", "Chevrolet", "Ford", "Hyundai", "Buick", "Jeep", "Mitsubishi", "Mazda", 
        "Volkswagen", "Toyota", "Subaru", "Dodge", "FIAT", "Chrysler", "Honda", "GMC", "Ram", "Pontiac", 
        "Saturn", "Mercury", "Saab", "Suzuki", "Oldsmobile", "Isuzu", "Plymouth", "Geo", "MINI", "smart", 
        "Scion", "HUMMER", "Fisker", "Daewoo", "Lotus", "Ferrari", "Lamborghini"
    ]

    styles = ['cab', 'Cab', 'van', 'Van', 'crew', 'Crew', 'koup', 'Koup', 'SUV', 'suv',
            'convertible', 'Convertible', 'Wagon', 'wagon', 'Coupe', 'coupe', 'Hatchback',
            'hatchback', 'sedan', 'Sedan']

    ultra_luxury_brands = ["Bentley", "Rolls-Royce"]

    high_end_luxury_brands = ["Aston Martin", "Maserati"]

    luxury_brands = ["Audi", "BMW", "Cadillac", "Jaguar", "Land Rover", "Lexus", "Lincoln",
                    "Mercedes-Benz", "Volvo", "Acura", "Infiniti"]
    premium_performance_and_sports_brands = ["Porsche", "Tesla", "Lotus", "Ferrari", "Lamborghini"]

    mainstream_brands = ["Kia", "Nissan", "Chevrolet", "Ford", "Hyundai", "Buick", "Jeep",
                        "Mitsubishi", "Mazda", "Volkswagen", "Toyota", "Subaru", "Dodge",
                        "FIAT", "Chrysler", "Honda", "GMC", "Ram", "MINI",  "HUMMER"]

    economy_and_budget_friendly_brands = ["Pontiac", "Saturn", "Mercury", "Oldsmobile", "Saab", "Suzuki",
                                        "Isuzu", "Plymouth", "Geo", "smart", "Scion", "Fisker", "Daewoo"]
    for name in styles:
        mask = df['body'].str.contains(name)
        if name in ['cab', 'Cab', 'crew', 'Crew']: 
            df.loc[mask, 'body'] = 'Truck'
        elif name in ['koup', 'Koup', 'Coupe', 'coupe']:
            df.loc[mask, 'body'] = 'Coupe'
        elif name in ['SUV', 'suv']:
            df.loc[mask, 'body'] = 'SUV'
        elif name in ['convertible', 'Convertible']:
            df.loc[mask, 'body'] = 'Convertible'
        elif name in ['Wagon', 'wagon']:
            df.loc[mask, 'body'] = 'Wagon'
        elif name in ['van', 'Van']:
            df.loc[mask, 'body'] = 'Van'
        elif name in ['sedan', 'Sedan']:
            df.loc[mask, 'body'] = 'Sedan'
        elif name in ['hatchback', 'Hatchback']:
            df.loc[mask, 'body'] = 'Hatchback'

    for brand in ultra_luxury_brands + high_end_luxury_brands + luxury_brands + \
                premium_performance_and_sports_brands + mainstream_brands + \
                economy_and_budget_friendly_brands:
        mask = df['make'].str.contains(brand, case=False, na=False)
        if brand in ultra_luxury_brands:
            df.loc[mask, 'category'] = 'Ultra Luxury'
        elif brand in high_end_luxury_brands:
            df.loc[mask, 'category'] = 'High End Luxury'
        elif brand in luxury_brands:
            df.loc[mask, 'category'] = 'Luxury'
        elif brand in premium_performance_and_sports_brands:
            df.loc[mask, 'category'] = 'Performance'
        elif brand in mainstream_brands:
            df.loc[mask, 'category'] = 'Mainstream'
        elif brand in economy_and_budget_friendly_brands:
            df.loc[mask, 'category'] = 'Economy / Budget'
    df.to_csv('cleaned_final_data.csv', index=False) 

# Log transforms the odometer (milage) attribute for more accurate predictions in model
def step_three():
    df = pd.read_csv('cleaned_final_data.csv')
    column = 'odometer'
    df[column] = StandardScaler().fit_transform(np.array(df[column]).reshape(-1, 1))
    df.to_csv('LogTransformed_final_data.csv', index=False)

if __name__ == "__main__":
    dataframe = step_one()
    step_two(dataframe)
    step_three()