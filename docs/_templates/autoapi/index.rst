Installation
============

.. code-block:: bash

   pip install step-kit

Then install dgl for your specific environment. For example, to install dgl 1.1.3 with cuda 11.7, which is the version used in the tutorials, you can run the following command:

.. code-block:: bash

   pip install dgl==1.1.3 -f https://data.dgl.ai/wheels/cu117/repo.html

dgl official installation guide can be found `here <https://docs.dgl.ai/install/index.html>`_.

Tutorials
=========
This page contains tutorials on how to use the STEP.

.. toctree::
   :titlesonly:
   :maxdepth: 1

   ../../notebooks/DLFPC
   ../../notebooks/Human_Lymph_Node
   ../../notebooks/MERFISH
   ../../notebooks/scRNA-seq
   ../../notebooks/human_colorectal_cancer
   ../../notebooks/mouse_small_intestine

API Reference
=============

This page contains auto-generated API reference documentation.

.. toctree::
   :titlesonly:

   {% for page in pages %}
   {% if page.top_level_object and page.display %}
   {{ page.include_path }}
   {% endif %}
   {% endfor %}