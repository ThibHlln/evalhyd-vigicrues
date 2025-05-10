Un add-on Python pour évaluer les prévisions VigiCrues avec EvalHyd
===================================================================

``evalhyd.vigicrues`` est un module Python complémentaire à ``evalhyd``
proposant un pré-traitement et un post-traitement spécifique aux
`prévisions hydrologiques du réseau VigiCrues`_. En particulier, ce module permet
de lire des données d'observations (fichiers XML-SANDRE ou CSV) et de
prévisions (fichiers XML-SANDRE ou PRV) et de les convertir au format
attendu par ``evalhyd``. Les données ainsi converties peuvent ensuite être
évaluées de façon déterministe ou probabiliste via des fonctions
spécifiques qui font des liens directs avec celles d'``evalhyd``. Des
fonctions de visualisation complètent ce module (histogrammes de rangs,
diagrammes de fiabilité).

.. _prévisions hydrologiques du réseau VigiCrues: https://www.vigicrues.gouv.fr

.. grid:: 1 1 5 5
   :gutter: 2

   .. grid-item-card::
      :text-align: center

      :fas:`rocket;sd-text-info fa-4x`

      .. button-ref:: installation
         :ref-type: doc
         :color: primary
         :click-parent:
         :outline:

         Installation

   .. grid-item-card::
      :text-align: center

      :fas:`school;sd-text-info fa-4x`

      .. button-ref:: tutoriel
         :ref-type: doc
         :color: primary
         :click-parent:
         :outline:

         Tutoriel

   .. grid-item-card::
      :text-align: center

      :fas:`gear;sd-text-info fa-4x`

      .. button-ref:: api
         :ref-type: doc
         :color: primary
         :click-parent:
         :outline:

         Référence API

   .. grid-item-card::
      :text-align: center

      :fas:`code;sd-text-info fa-4x`

      .. button-link:: https://gitlab.irstea.fr/HYCAR-Hydro/evalhyd/evalhyd-vigicrues
         :color: primary
         :click-parent:
         :outline:

         Code source :fas:`arrow-up-right-from-square`

   .. grid-item-card::
      :text-align: center

      :fas:`bug;sd-text-info fa-4x`

      .. button-link:: https://gitlab.irstea.fr/HYCAR-Hydro/evalhyd/evalhyd-vigicrues/-/issues
         :color: primary
         :click-parent:
         :outline:

         Rapport de bugs :fas:`arrow-up-right-from-square`


.. note::

   Ce module a été développé grâce au soutien du `Service Central
   VigiCrues`_ dans le cadre de sa convention avec l'INRAE.

   .. _Service Central VigiCrues: https://lannuaire.service-public.fr/gouvernement/7e9f75fc-5862-4a33-8015-da82e7edfafd

   .. figure:: ../__images/vigicrues-logo.svg
      :width: 300px
      :align: center
      :alt: logo VigiCrues
      :target: https://www.vigicrues.gouv.fr



.. toctree::
   :hidden:
   :maxdepth: 1

   installation
   tutoriel.ipynb
   api
   licence
