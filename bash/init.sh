export NETPATH=$(dirname "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )")

export PYTHONPATH=$NETPATH/anaconda3/lib/python3.6/site-packages
PYTHONPATH=$PYTHONPATH:$NETPATH/Network/python

PATH=$PATH:$NETPATH/Network/scripts
