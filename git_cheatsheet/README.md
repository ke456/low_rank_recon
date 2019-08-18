# Simple git workflow cheatsheet
## Simple concepts
### Master vs Topic branches
The idea of git is to provide a simple workflow paradigm. The "master" branch is the "source of truth", whereas, a "topic" branch is the branch used for progress and experimentation. The reason why we have master and topic branches is so that we can do a "pull review", prior to merging any changes into master.

"Pull reviews" are simple: Another developer who is related to the change you want to merge, can review the change and comment on it, prior to being merged in. The benefit of this is that we are not pushing changes into master that isn't reviewed, or in otherwords, rough changes that is likely be removed. Think of it as getting a pair of eyes to look over an essay before you submit it.

Topic branches also provide a frozen version of master. As soon as you make the topic branch based off the current master, the topic branch diverts from master and your changes will not be affected by any change from master. If there is a conflict in files change, git will tell you by throwing a "merge conflict" during a pull review. 

### Remote vs Local
Remote is the version that is stored on github. Local is the version that is stored on your computer.

## Creating a topic branch
First thing you want to do it update your local master with the remote master.
```
git checkout master
git fetch
git rebase origin master
```
Next, we can create a new branch based off the updated local master.
```
git checkout -b <topicbranch_name>
```
That's it!

## Updating your topic branch with the latest master
Sometimes you want to make sure your topic branch is similar to the master branch. You can do this by moving all the commits on your topic branch onto the latest copy of your master branch with the following code:
```
# Switch to master branch
git checkout master
# Fetch the latest changes from remote master
git fetch
# Update the master HEAD to point at the remote master HEAD
git rebase origin master
# Switch back to topic branch
git checkout <topicbranch_name>
# Move all the commits from the topic branch to be ontop of the master HEAD
git rebase master
```

## Making a pull request
To make a pull request, you MUST do it from Github UI. you can do it from accessing your branch in Github. Make sure to tag me with the @ symbol.
