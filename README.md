# mixteclabeling
Repository with directions, label, scripts, and information for mixtec labeling metadata.

## Segmenting Workflow:

# Load codex page into Segment-Anything Demo:

![Select page](tutorial_images/1_select_page.png)

# Add sufficient number of points (mask prompts) to segment figure
![Add or remove areas to mask](tutorial_images/2_add_masks.png)

# Cut-out segmented figure 
![Cut-out figure](tutorial_images/3_cutout_figure.png)

# Drag figure to "cutouts" folder to export
![Export figure](tutorial_images/4_drag_figure_cutouts.png)

# Rename newly exported png figure according to this scheme:
![Rename figure](tutorial_images/5_rename_figure.png)

* "pageNumber-qualityIndicator-figureOnPageCount.png"

	* Page numbers are relative to each codex and can be found in the file name for the given page. 

	* Qualities range from a to c, and examples can be found below:
		* A. ![](tutorial_images/a_example.png)
		* B. ![](tutorial_images/b_example.png)
		* C. ![](tutorial_images/c_example.png)

	* The last two-digit number in the file name corresponds the order in which the figures are segmented. 

* For example:

	* "001-a-01.png" -> This stands for the first page (001), "a" level quality, and the first figure segmented on the given page (01).

* Accordingly, the second figure encountered on the first page will be (assuming "a" level quality): 
	* "001-a-02.png"

* If the figure extracted is an animal, or non-human entity, it will be named similarly, except we use a character from the alphabet instead of a two-digit number (assuming "a" level quality:

	* "001-a-a.png"

* Accordingly, the second animal, or non-human entity, encountered on the first page will be named (assuming "a" level quality):

	* "001-a-b.png"

