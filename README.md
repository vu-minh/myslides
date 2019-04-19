# talk-template

A ready-to-fork template for talks, using [remark](https://github.com/gnab/remark), [KaTeX](https://github.com/Khan/KaTeX) and some customised CSS.

# export to pdf

```
(16:16):myslides$ decktape http://127.0.0.1:5500/?p=esann2019.md pdf/esann2019.pdf
Loading page http://127.0.0.1:5500/?p=esann2019.md ...
Live reload enabled.
Loading page finished with status: 200
Remark JS plugin activated
Printing slide #12      (12/12) ...
Printed 12 slides
```


## Instructions

- Clone this repository:
```
git clone https://github.com/glouppe/talk-template.git
cd talk-template
```
- Start an HTTP server to serve the slides:
```
python -m http.server 8001
```
(Or `http-server -p 8001` if having [http-server](https://github.com/indexzero/http-server) installed)
- For live reload feature, use nodejs [live-server](https://github.com/tapio/live-server)
(Or `live-server` in vscode with default port 5500)

- Edit `talk.md` for making your slides.
- Use [decktape](https://github.com/astefanutti/decktape) for exporting your slides to PDF.

## Markup language

Slides are written in Markdown. See the remark [documentation](https://github.com/gnab/remark/wiki/Markdown) for further details regarding the supported features.

This template also comes with grid-like positioning CSS classes (see `assets/grid.css`) and other custom CSS classes (see `assets/style.css`)

## Integration with GitHub pages

Slides can be readily integrated with [GitHub pages](https://pages.github.com/) by hosting the files in a GitHub repositery and enabling Pages in the Settings tab.

See e.g. [https://glouppe.github.io/talk-template](https://glouppe.github.io/talk-template) for this deck. 
