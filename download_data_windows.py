import urllib.request
import os

# 1. Create the folder to hold the files
folder_name = "nyc_taxi_data"
os.makedirs(folder_name, exist_ok=True)

# 2. Modern dataset years (2023-2025)
years = ['2023', '2024', '2025']
months = [f"{str(i).zfill(2)}" for i in range(1, 13)]

print("Starting resilient download of NYC Taxi Data...")
print("Since files are downloaded one by one, your laptop won't crash if the Wi-Fi drops!\n")

for year in years:
    for month in months:
        file_name = f"yellow_tripdata_{year}-{month}.parquet"
        url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{file_name}"
        save_path = os.path.join(folder_name, file_name)
        
        # Check if file already exists so we don't waste your bandwidth!
        if os.path.exists(save_path):
            print(f"⏭️ {file_name} already exists. Skipping...")
            continue
            
        print(f"⬇️ Attempting to download {file_name} (Usually ~45MB)...")
        try:
            urllib.request.urlretrieve(url, save_path)
            print(f"   ✅ Success!")
        except urllib.error.HTTPError:
            print(f"   ❌ File not found on server (NYC Govt hasn't uploaded it yet).")
        except Exception as e:
            print(f"   ⚠️ Wi-Fi error while downloading {file_name}: {e}")

print("\n🎉 Download process complete! You can now run your Streamlit app.")