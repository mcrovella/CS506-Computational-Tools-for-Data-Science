##############################
# Originally from https://github.com/jappavoo/UndertheCovers
# As of 8/25/2021
# Jappavoo's extensive tools for customizing and extending jupyter for lecturing
##############################

# imports to make python code easier and constent
from IPython.core.display import display, HTML, Markdown, TextDisplayObject, Javascript
from IPython.display import IFrame, Image
import ipywidgets as widgets
from ipywidgets import interact, fixed, Layout
import os, requests
from notebook.notebookapp import list_running_servers
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import pandas as pd

matplotlib.rcParams['animation.html'] = 'jshtml'

#from IPython.display import Javascript

#var ipkernel = IPython.notebook.kernel;
#var stringHostName = window.location.hostname
#var ipcommand = "NB_HOST = " + "'"+stringHostName+"'";
#ipkernel.execute(ipcommand);
#"""
#
#display(Javascript(js_code))

# common functions 
# Custom functions and classes to help with standard slide elements that I use
# NOTE:  I don't know python so this probably all should be rewritten by someone
#        who knows what they are doing.  

# This probably should be a class
def MkCodeBox(file, lang, html_title, w, h):
    #open text file in read mode
    text_file = open(file, "r")
    #read whole file to a string
    data = text_file.read()
    #close file
    text_file.close()
    # build contents from file and language
    md_text = '''
``` ''' + lang + '''
''' + data + '''
```
'''
    # build output widget 
    wout = widgets.Output(layout=Layout(overflow='scroll',
                                        width=w,
                                        min_width=w,
                                        max_width=w,
                                        min_height=h,
                                        height=h,
                                        max_height=h))
    with wout:
        display(Markdown(md_text),)
    display(HTML(html_title))
    return wout

# Make a box that display the specified image files from the specified
# directory if no files a specified then all file in the directory are
# displayed.  A slider is used to select between the images
# Note if you want control the order you must specifiy the files
# explicitly
#  This function requires backend kernel see 
def mkImgsBox(dir,files=[]):
    if len(files)==0:
        files=os.listdir(dir);
    interact(lambda i,d=fixed(dir),
             f=fixed(files): 
             display(Image(dir + '/' + files[i])),
             i=widgets.IntSlider(min=0, max=(len(files)-1), step=1, value=0));

def files_to_imgArray(dir, files):
    n=len(files);
    imgs = [];
    for f in files:
        imgs.append(plt.imread(dir + "/" + f))
    return imgs;

# this embeddes a javascript animation box with specified
# images
def mkImgsAnimateBox(dir, files ,dpi=100.0,xpixels=0,ypixels=0):
    imgs=files_to_imgArray(dir, files)
    if (xpixels==0):
        xpixels = imgs[0].shape[0]
    if (ypixels==0):
        ypixels = imgs[0].shape[1]
    fig = plt.figure(figsize=(ypixels/dpi, xpixels/dpi), dpi=dpi)
    fig.patch.set_alpha(.0)
    im = plt.figimage(imgs[0])
    def animate(i):
        im.set_array(imgs[i])
        return(im,);
    ani=animation.FuncAnimation(fig, animate, frames=np.arange(0,(len(imgs)-1),1), fargs=None, interval=100, repeat=False)
    # next line is used to remove side affect of plot object
    plt.close()
    return ani

# for future reference incase we want to move to a plotly express version
# import plotly.express as px
# import matplotlib.pyplot as plt
# import numpy as np
# from skimage import io
# imgs = files_to_imgArray("../images/UnixL01_SHCHT1", [
#     '05SHLLChat.png',
#     '06SHLLChat.png',
#     '07SHLLChat.png',
#     '08SHLLChat.png',
#     '09SHLLChat.png',
#     '10SHLLChat.png',
#     '11SHLLChat.png',
#     '12SHLLChat.png',
#     '13SHLLChat.png',
#     '14SHLLChat.png',
#     '15SHLLChat.png',
#     '16SHLLChat.png',
#     '17SHLLChat.png',
#     '20SHLLChat.png']);

#plt.imshow(imgs[0])
#px.imshow(imgs[0])
#fig = px.imshow(np.array(imgs), animation_frame=0, labels=dict(), height=(), width=())
#fig.update_xaxes(showticklabels=False) \
#    .update_yaxes(showticklabels=False)

# binary and integer utilities
def bin2Hex(x, sep=' $\\rightarrow$ '):
    x = np.uint8(x)
    md_text="0b" + format(x,'08b') + sep + "0x" + format(x,'02x')
    display(Markdown(md_text))


# this displays a table of bytes as binary you can pass a label array if you want row labels
# for the values.  You can also override the column labels if you want.  Not sure if what
# I did for controlling centering is the best but it works ;-)
# probably want to make more of the styling like font size as parameters
#
# examples:
#   Simple use cases are to display a single value in various formats
#    displayBytes([0xff])
#    displayBytes([0xff],columns=[])
#    displayBytes([0xff],labels=["ALL ON"])
#    displayBytes([0xff],columns=[],labels=["ALL ON"])
#
#   Using to show a simple multi value  example:
#    u=np.uint8(0x55)
#    v=np.uint8(0xaa)
#    w=np.bitwise_and(u,v)
# Empty value is indicated via [""] note also force cell height to avoid
#    the empty row from shrinking 
#    displayBytes([[u],[v],[""]],labels=["u","v", "u & v"], td_height="85px")
#    displayBytes([[u],[v],[w]],labels=["u","v", "u & v"])
#
# Some tablen example
#    ASCII table
#    displayBytes(bytes=[[i] for i in range(128)], labels=["0x"+format(i,"02x")+ " (" + format(i,"03d") +") ASCII: " + repr(chr(i)) for i in range(128)])
#    Table of all 256 byte values
#    displayBytes(bytes=[[i] for i in range(256)], labels=["0x"+format(i,"02x")+ " (" + format(i,"03d") +")" for i in range(256)], center=True)
def toBits(v,dtype,count,numbits):
    try:
        x=np.unpackbits(dtype(v),count=count)
        if (numbits<8):
            return x[numbits:]
        else:
            return x
    except:
#        print("oops v: ", v, type(v), len(v));
        return [" " for i in range(numbits)]
        
def displayBytes(bytes=[[0x00]],
                 labels=[],
                 labelstitle="",
                 prefixvalues=[],
                 prefixcolumns=[],
                 numbits=8,
                 dtype=np.uint8,
                 columns=["[$b_7$","$b_6$", "$b_5$", "$b_4$", "$b_3$", "$b_2$", "$b_1$","$b_0$]"],
                 center=True,
                 th_font_size="1.5vw",
                 td_font_size="3vw",
                 td_height="",
                 border_color="#cccccc",
                 tr_hover_bgcolor="#11cccccc",
                 tr_hover_border_color="red",
                 td_hover_bgcolor="#880000",
                 td_hover_color="white"
                 ):

    # if no labels specified then send in blanks to supress
    # there is probably a better way to do this
    #if not labels:
    #    labels = ["" for i in range(len(bytes))]

    sizeinbits = (dtype(0).nbytes)*8

    # have attempted to support specifiy the number of bits
    # but not sure it really works will need to be tested
    if numbits<sizeinbits:
        count=sizeinbits;
    else:
        count=numbits
        
    # convert each byte value into an array of bits        
    try:    
        x = np.unpackbits(np.array(bytes,dtype=dtype),count,axis=1)
        if (numbits<sizeinbits):
            x = [ i[numbits:] for i in x ]
    except:
        x = np.array([ toBits(i,dtype=dtype,count=count,numbits=numbits) for i in bytes ])
        
    # Add any prefix data columns to the bits 
    if prefixvalues:
        x = np.concatenate((prefixvalues,x),axis=1)
            
    if not columns:
        if not labels:
            df=pd.DataFrame(x)
        else:
            df=pd.DataFrame(x,index=labels)
    else:
        # if extra prefix column labels specified then add them
        # to the front of the other column labels
        if prefixcolumns:
            columns=np.concatenate((prefixcolumns,columns))
        if not labels:
            df=pd.DataFrame(x,columns=columns)
        else:
            df=pd.DataFrame(x,index=labels,columns=columns)
            
    # style the table
    if labelstitle:
        df = df.rename_axis(labelstitle, axis="columns")
        
    th_props = [
        ('font-size', th_font_size),
        ('text-align', 'center'),
        ('font-weight', 'bold'),
        ('color', 'white'),
      ('background-color', 'black')
    ]
    td_props = [
            ('border','4px solid ' + border_color),
            ('font-size', td_font_size),
#            ('color', 'white'),
            ('text-align', 'center'),
#            ('background-color', 'black'),
            ('overflow-x', 'hidden')
    ]
    if td_height:
        # print("adding td_height: ", td_height)
        td_props.append(('height', td_height))
        
    td_hover_props = [
        ('background-color', td_hover_bgcolor),
        ('color', td_hover_color)
    ]
    tr_hover_props = [
        ('background-color', tr_hover_bgcolor),
        ('border', '4px solid ' + tr_hover_border_color)
    ]

    body=df.style.set_table_styles([
            {'selector' : 'td', 'props' : td_props },
            {'selector' : 'th', 'props': th_props },
            {'selector' : 'td:hover', 'props': td_hover_props },
            {'selector' : 'tr:hover', 'props': tr_hover_props }
        ])
    
    # if no row labels hide them
    if (len(labels)==0):
        body.hide_index()
    # if no column labels hide them 
    if (len(columns)==0):
        body.hide_columns()
        
    # make body sticky header if present stay in place    
    body.set_sticky(axis=1)

    # center in frame
    if center:
        margins=[
            ('margin-left', 'auto'),
            ('margin-right', 'auto')
            ]
        body.set_table_styles([{'selector': '', 'props' : margins }], overwrite=False);
    body=body.to_html()
    display(HTML(body))   

def mkHexTbl():
    displayBytes(bytes=[i for i in range(16)], numbits=4, 
                 prefixvalues=[[format(i,"0d"),format(i,"1x")] for i in range(16)],
                 prefixcolumns=["Dec", "Hex"],
             columns=["[$b_3$", "$b_2$", "$b_1$", "$b_0$]"])

def displayStr(str, size="", align=""):
    md='<div'
    if size:
        md = md + ' style="font-size: ' + size + ';"'
    if align:
        md = md + ' align="' + align + '"'
    md = md + ">\n"
    md = md + str
    md = md + "\n</div>"
#    print(md)
    return display(Markdown(md))

# print("Common executed")
