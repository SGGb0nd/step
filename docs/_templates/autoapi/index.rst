Installation
============

.. code-block:: bash

    pip install step-kit

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