import pandas as pd
import matplotlib.pyplot as plt
from fredapi import Fred

# You'll need to get an API key from FRED
fred = Fred(api_key='YOUR_API_KEY')

# Fetch data for GDP and Unemployment Rate
gdp = fred.get_series('GDP')
unemployment = fred.get_series('UNRATE')

# Create a simple plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

gdp.plot(ax=ax1)
ax1.set_title('US GDP')
ax1.set_ylabel('Billions of Dollars')

unemployment.plot(ax=ax2)
ax2.set_title('US Unemployment Rate')
ax2.set_ylabel('Percent')

plt.tight_layout()
plt.show()