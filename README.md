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
+ First, clone repository:
```
git clone https://github.com/jpcadena/classification-tweets-national-security-ecuador.git
```

+ Change directory to root project with:
```
  cd classification-tweets-national-security-ecuador
```

+ Create your git branch with the following:
```
git checkout -b <new_branch>
```

For *<new_branch>* use some convention as following:
```
yourgithubusername
```

Or if some work in progress (WIP) or bug shows up, you can use:
```
yourgithubusername_feature
```

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
git commit -m "Commit message"
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

+ Create a **virtual environment** 'sample_venv' with:
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


### Installation of libraries and dependencies
```
pip install -r requirements.txt
```


### Execution
```
python main.py
```


### Environment credentials
Rename **sample.env** to **.env** and replace your Twitter API credentials if you will work with **Tweepy** data collection.\
Ask for **Elevated** privileges for your developer account which is required for **Twitter API v2** and it takes around 48 hours to be accepted. Check more info at [Twitter Developers](https://developer.twitter.com/en).


### Documentation
Use docstrings with **reStructuredText** format by adding triple double quotes **"""** after function definition.\
Add a brief function description, also for the parameters including the return value and its corresponding data type.


### Additional information
If you want to give more style and a better format to this README.md file, check documentation at [Github Docs](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).\
Please use **linting** to check your code quality following [PEP 8](https://peps.python.org/pep-0008/). Check documentation for [Visual Studio Code](https://code.visualstudio.com/docs/python/linting#_run-linting) or for [Jetbrains Pycharm](https://github.com/leinardi/pylint-pycharm/blob/master/README.md).\
Recommended plugin for autocompletion: [Tabnine](https://www.tabnine.com/install)
