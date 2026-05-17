Jansen Linkages are complex mechanisms sensitive to small changes, and given that many hobbyists attempt to build scale model
in media that are highly bound to integer proportions such as Minecraft and Lego I decided to make a tool to help builders find
the right measurements for their builds

This project contains the following tools:
  Webapp (index.html)- Visualise the path of the linkage and dynamically modify the lengths. Has built in integer scaling.
  optimize_integer.py - Enter the scale, ex: --scale 0.2 to get top X (--top 3) configurations for best performing integer measurements.
                           After the script runs it will generate local urls that you can paste in your browser to see the optimal measurement
                           linkages in the web app
                           
  Gif - Makes a gif of the linkage over 720 degrees. Takes a long time. Values are hard coded. Go into the sctipt and change them and then run the script.

Usage:
  *  Just copy everything on your computer and maintain the folder structure.
  *  Double click on index.html to launch the webapp in your browser.
  *  For the IntegerOptimization script, launch a command prompt in the folder with the script and type  optimize_integer.py --scale<your scale>
     so if your scale is 0.2, do optimize_integer.py --scale 0.2. There are also other flags that you can run the script with. By default the top 3
     configurations are shown
