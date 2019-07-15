export NETPATH=$(dirname "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )")

export PYTHONPATH=$NETPATH/anaconda3/lib/python3.7/site-packages
PYTHONPATH=$PYTHONPATH:$NETPATH/python

PATH=$PATH:$NETPATH/bin
