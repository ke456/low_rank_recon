from train_select import train_select

# The list of tests which we will retrain or tune on.
tests = ['hcv', 'liver', 'survey', 'thyroid']

def train_all():
    for test in tests:
        train_select(test)
        
train_all()