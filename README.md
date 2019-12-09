# web-flyover

This app provides an graph interface to the network of all the hyperlinks between websites ending in ".gov", as scraped by 
[Common Crawl](https://commoncrawl.org/2019/11/host-and-domain-level-web-graphs-aug-sep-oct-2019/).  
The center node is the current website, whose title is at the top. On the left hand side are
the sites most representative of those that link to it, and on the right are those most representative
of those it links to. The sensitivity, which you can adjust with the slider, is what calibrates this "representativeness:" the higher the sensitivity, the more 
idiosyncratic representatives you will get. The lower the sensitivity, the more the sites will be representative 
of the network as a whole.  
When you click on a node for a site in the graph, it will take you to that site's graph. You can in this way explore a path of relatedness,
as well as find your way back with the back button. If you are curious about what any of the sites are, the legends on the right and
left are hyperlinked. If you have a specific site in mind you would like to start from, you can search for it in the dropdown.  
The edge thicknesses indicate the strength of the relatedness between two nodes.  
Enjoy. If you would like to share any ideas or experiences, please find me at trent wat son 1 at gmail.
