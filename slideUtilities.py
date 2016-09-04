from IPython.display import display_html
from IPython.display import display
import matplotlib.pyplot as plt
from IPython.display import HTML
import binascii
import base64

def hide_code_in_slideshow():
    import os
    uid = binascii.hexlify(os.urandom(8)).decode('UTF-8')
    html = """<div id="%s"></div>
    <script type="text/javascript">
        $(function(){
            var p = $("#%s");
            if (p.length==0) return;

            while (!p.hasClass("cell")) {
                p=p.parent();

                if (p.prop("tagName") =="body") return;
            }
            var cell = p;
            cell.find(".input").addClass("hide-in-slideshow")
        });
    </script>""" % (uid, uid)
    display_html(html, raw=True)
    return html

##########################
# python notebook does not support matplotlib animations
# so these functions create an .mp4 video and then display
# it using inline HTML
##########################

# Source: http://nbviewer.ipython.org/url/jakevdp.github.io/downloads/notebooks/AnimationEmbedding.ipynb


# updated to python3, now uses package base64 to do the encoding, reading comments from this post:
# http://jakevdp.github.io/blog/2013/05/12/embedding-matplotlib-animations/
from tempfile import NamedTemporaryFile

VIDEO_TAG = """<video controls>
 <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
 Your browser does not support the video tag.
</video>"""

def anim_to_html(anim):
    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.mp4') as f:
            anim.save(f.name, fps=20, extra_args=['-vcodec', 'libx264'])
            video = open(f.name, "rb").read()
        anim._encoded_video = base64.b64encode(video).decode('utf-8')
    
    return VIDEO_TAG.format(anim._encoded_video)

def display_animation(anim):
    # plt.close(anim._fig)
    return HTML(anim_to_html(anim))

def display_saved_anim(fname):
    with open(fname,'rb') as f:
        video = f.read()
    return HTML(VIDEO_TAG.format(base64.b64encode(video).decode('utf-8')))
