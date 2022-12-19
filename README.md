# classification-tweets-national-security-ecuador

## Classification of Tweets about national security at Ecuador 2022
Lorem Ipsum is simply dummy text of the printing and typesetting industry.
Lorem Ipsum has been the industry's standard dummy text ever since the 1500s,
when an unknown printer took a galley of type and scrambled it to make a type
specimen book. It has survived not only five centuries, but also the leap into
electronic typesetting, remaining essentially unchanged. It was popularised in
the 1960s with the release of Letraset sheets containing Lorem Ipsum passages,
and more recently with desktop publishing software like Aldus PageMaker
including versions of Lorem Ipsum.

### Requirements
Python 3.10+

### Git
+ Clone repository:
```
git clone https://github.com/jpcadena/classification-tweets-national-security-ecuador.git
```

+ Change directory to root project with:

```
  cd classification-tweets-national-security-ecuador
```

+ First create your git branch with the following:
```
git checkout -b <new_branch>
```

For *<new_branch>* use some convention as following:
- **yourgithubusername**

If some work in progress (WIP) or bug shows up, **yourgithubusername_feature**

+ Switch to your branch:
```
git checkout <new_branch>
```

+ Add your new files and changes:
```
git add .
```

+ Make your commit with a reference message about the fix/changes.
```
git commit -m "Message commit"
```

+ First push for remote branch:
```
git push --set-upstream origin <new_branch>
```

+ Latter pushes:
```
git push origin
```

### Environment

+ Create a **virtualenvironment** 'sample_venv' with:

```
python3 -m venv sample_venv
```

+ Activate environment in Windows with:

```
.\sample_venv\Scripts\activate
```

+ Or with Unix or Mac:

```
source sample_venv/bin/activate
```

### Installation

```
pip install -r requirements.txt
```

### Execution

```
python main.py
```

### Environment credentials

Rename **sample.env** to **.env** and replace your Twitter API credentials if
you will work with **Tweepy** data collection.
Ask for **Elevated** privileges for your developer account which is required
for **Twitter API v2**. Check more info
at [Twitter Developers](https://developer.twitter.com/en/portal/dashboard).


### Additional information
If you want to give more to this README.md file, check documentation at [Github Docs](https://docs.github.com/es/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).
