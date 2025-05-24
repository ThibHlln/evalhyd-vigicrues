.. currentmodule:: evalhyd.vigicrues

Référence API
=============

Lecture
-------

.. if-builder:: html

   .. autosummary::
      :template: function.rst
      :toctree: ./api/functions

      read_obs_from_xml_sandre
      read_obs_from_csv_hydroportail

      read_prd_from_xml_sandre
      read_prd_from_prv

.. if-builder:: simplepdf

   .. autofunction:: read_obs_from_xml_sandre
   .. autofunction:: read_obs_from_csv_hydroportail
   .. autofunction:: read_prd_from_xml_sandre
   .. autofunction:: read_prd_from_prv


Évaluation
----------

.. if-builder:: html

   .. autosummary::
      :template: function.rst
      :toctree: ./api/functions

      evalp
      evald

.. if-builder:: simplepdf

   .. autofunction:: evalp
   .. autofunction:: evald


Visualisation
-------------

.. if-builder:: html

   .. autosummary::
      :template: function.rst
      :toctree: ./api/functions

      plot_rank_hist
      plot_rel_diag

.. if-builder:: simplepdf

   .. autofunction:: plot_rank_hist
   .. autofunction:: plot_rel_diag
