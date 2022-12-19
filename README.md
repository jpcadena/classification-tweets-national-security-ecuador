# classification-tweets-national-security-ecuador

## Classification of Tweets about national security at Ecuador 2022

### Environment credentials

Rename **sample.env** to **.env** and replace your Twitter API credentials if you
will work with *Tweepy* data collection.
Ask for **Elevated** privileges for your developer account which is required for
**Twitter API v2**. Check more info
at [Twitter Developers](https://developer.twitter.com/en/portal/dashboard).

### Environement and execution

+ Create a **virtualenvironment** 'sample_venv' with:
```
python3 -m venv sample_venv
cd classification-tweets-national-security-ecuador
```

+ Activate environment in Windows with:
```
.\sample_venv\Scripts\activate
```
+ Or with Unix or Mac:
```
source sample_venv/bin/activate
```

+ Install dependencies with:
```
pip install -r requirements.txt
```

+ Execute main script with:
```
python main.py
```

PD:
If you want to give more style for this README.md file, check documentation at [Github Docs](https://docs.github.com/es/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).
