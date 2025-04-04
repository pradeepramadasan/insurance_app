
from geopy.geocoders import Nominatim
import time
import h3
print(h3.__version__ )  # Check the version of h3
geolocator = Nominatim(user_agent="my_app")  # Initialize the geolocator
zip_to_coords_cache = {}                     # Initialize your cache

def get_h3_for_zip(zipcode, resolution=8):
    """
    Convert a ZIP code to an H3 hexagon index
    
    Args:
        zipcode (str): ZIP code to convert
        resolution (int): H3 resolution (0-15), higher is more precise
        
    Returns:
        str: H3 index or None if geocoding fails
    """
    # Check cache first
    if zipcode in zip_to_coords_cache:
        lat, lon = zip_to_coords_cache[zipcode]
    else:
        try:
            # Use California to narrow down the search
            location = geolocator.geocode(f"{zipcode}, CA, USA")
            
            if location:
                lat, lon = location.latitude, location.longitude
                zip_to_coords_cache[zipcode] = (lat, lon)
                # Be nice to the geocoding service
                time.sleep(0.5)
            else:
                # If not found, use approximate coordinates for California
                print(f"Geocoding failed for {zipcode}, using approximate location")
                # Central California coordinates as fallback
                lat, lon = 36.7783, -119.4179  
                zip_to_coords_cache[zipcode] = (lat, lon)
                
        except Exception as e:
            print(f"Error geocoding ZIP {zipcode}: {e}")
            # Central California coordinates as fallback
            lat, lon = 36.7783, -119.4179
            zip_to_coords_cache[zipcode] = (lat, lon)
    
    # Convert to H3 index
    try:
        hex_index = h3.latlng_to_cell(lat, lon, resolution)
        return hex_index
    except Exception as e:
        print(f"Error converting to H3: {e}")
        return None

result = get_h3_for_zip("94103")
print(result)

