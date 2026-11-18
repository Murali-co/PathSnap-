"""Self-contained Folium vector map. Vendored Leaflet avoids runtime CDNs."""
from pathlib import Path
import folium
from branca.element import Element

ASSETS=Path(__file__).resolve().parent/'assets'


def make_map(result):
    g=result['geometry']
    valid=[p for p in g.get('gps',[]) if p is not None]
    center=[valid[len(valid)//2][1],valid[len(valid)//2][0]] if valid else [0,0]
    m=folium.Map(location=center,tiles=None,zoom_start=13,prefer_canvas=True)
    m.default_js=[];m.default_css=[]
    m.get_root().header.add_child(Element('<style>'+ (ASSETS/'leaflet.css').read_text()+'</style>'))
    m.get_root().header.add_child(Element('<script>'+ (ASSETS/'leaflet.js').read_text()+'</script>'))
    layers=[('Road network','#94a3b8',g.get('network',[]),1,True),
            ('GPS observations','#2563eb',g.get('gps_segments',[]),2,True),
            ('Classifier route','#f59e0b',g.get('classifier',[]),3,False),
            ('Decoded route','#059669',g.get('predicted',[]),5,True)]
    for name,color,lines,weight,show in layers:
        layer=folium.FeatureGroup(name=name,show=show)
        for line in lines:
            if len(line)>1:folium.PolyLine([[lat,lon] for lon,lat in line],color=color,weight=weight,opacity=.85,tooltip=name).add_to(layer)
            elif line:folium.CircleMarker([line[0][1],line[0][0]],radius=3,color=color).add_to(layer)
        layer.add_to(m)
    unmatched=folium.FeatureGroup(name='Unmatched GPS')
    for match,p in zip(result.get('matches',[]),g.get('gps',[])):
        if match['arc_id'] is None and p:folium.CircleMarker([p[1],p[0]],radius=4,color='#dc2626').add_to(unmatched)
    unmatched.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    if valid:m.fit_bounds([[min(p[1] for p in valid),min(p[0] for p in valid)],[max(p[1] for p in valid),max(p[0] for p in valid)]])
    m.get_root().html.add_child(Element('<div style="position:fixed;bottom:22px;left:12px;z-index:9999;background:white;padding:10px;border:1px solid #ccc;font:13px sans-serif">'
        '<b>PathSnap · vector road view</b><br><span style="color:#2563eb">● GPS</span> · <span style="color:#059669">━ Decoded route</span><br>'
        'Roads: supplied network · Leaflet © contributors</div>'))
    return m
