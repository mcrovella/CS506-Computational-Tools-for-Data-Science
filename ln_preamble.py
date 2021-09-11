##############################
# Originally from https://github.com/jappavoo/UndertheCovers
# As of 8/25/2021
# Jappavoo's extensive tools for customizing and extending jupyter for lecturing
##############################

# Assumes common.py
# NOTES:  My standard preamble ipython for lecture note rise slides
#    1) Note in terminals setting TERM=linux is friendly with emacs
#    2) we customize css to improve layout of cells in the browser window
#    3) standarize how to display code blocks from a source file
# 


### BOOTSTRAPPING CODE
# get server info so that we can make api calls when runing direclty on a
# jupyter notebook server
servers=list(list_running_servers())
# note that this assumes only one server is running on this host
# also that there is no left over garbage in Library/Jupyter/runtime
info=next(list_running_servers())
# localhost_url used for explicit calls to my server
localhost_url=info['url']
api_url=localhost_url + 'api/'
api_term_url=api_url + 'terminals'
api_token=info['token']
# urls used as relative to my server 
base_url=info['base_url']

# on the operate-firrst jupyterhub I found that api_token is not set but
# the JPY_API_TOKEN environment variable or JUPYTERHUB_API_TOKEN
# should exist
if not api_token:
    api_token = os.environ.get('JPY_API_TOKEN')

if not api_token:
    api_token = os.environ.get('JUPYTERHUB_API_TOKEN')
    
if not api_token:
    print("ERROR: unable to deterimine API token");


# get list of current terminals so that we can reuse this if enough exist 
# otherwise we will create new ones as needed
r=requests.get(url=api_term_url, headers={'Authorization': 'token ' + api_token})
TERMINALS=r.json()

def mkTerm():
    # create a terminal for our editor
    r=requests.post(url=api_term_url, headers={'Authorization': 'token ' + api_token})
    c=r.json()
    return c['name']

# assumes that we will use the first three terminals four our use
# create standard terminals for organizing editor, build and debugger
try:
    EDITORTERM=TERMINALS[0]['name']
except:
    EDITORTERM=mkTerm()

try:
    BUILDTERM=TERMINALS[1]['name']
except:
    BUILDTERM=mkTerm()

try:
    DEBUGGERTERM=TERMINALS[2]['name']
except:
    DEBUGGERTERM=mkTerm()


# FYI:  Don't need this yet so I am commenting out ...
#       It does work though
# hack to get the hostname of the current page
# could not figure out another way to get the
# hostname for embedded browser windows
# to use as their url

#from IPython.display import Javascript

js_code = """
var ipkernel = IPython.notebook.kernel;
var stringHostName = window.location.hostname
var ipcommand = "NB_HOST = " + "'"+stringHostName+"'";
ipkernel.execute(ipcommand);
"""
#
#display(Javascript(js_code))

# cusomization of ccs to make slides look better 

# MEC: removed for now - replace with per-notebook CSS in metadata
"""
display(HTML(
    '<style>'
        '#notebook { padding-top:0px !important; } ' 
        '.container { width:100% !important; } '
        '.CodeMirror { width:100% !important;}'
        '.end_space { min-height:0px !important; } '
        # '.prompt { display:none }'
        '.terminal-app #terminado-container { width:100%; }'
        'div.mywarn { background-color: #fcf2f2;border-color: #dFb5b4; border-left: 5px solid #dfb5b4; padding: 0.5em;}'
        'button { background-color: #cccccc00; }'
    '</style>'
))
"""

# show Terminal where TERMNAME is one of the terminals we created below
def showTerm(TERMNAME, title, w, h):
    if title:
        display(HTML('<b>' + title + '</b>'))
    return IFrame(base_url + 'terminals/' + TERMNAME, w,h)
    
def showET(title="TERMINAL Window for Editor"):
    return showTerm(EDITORTERM, title, "100%", 600)

def showBT(title="TERMINAL Window for Build Commands"):
    return showTerm(BUILDTERM, title, "100%", 200)

def showDT(title="TERMINAL Window for Debugger"):
    return showTerm(DEBUGGERTERM, title, "100%", 800)

# print("Preamble executed")
