happyrec.utils
==============

.. contents:: Contents
   :local:

Assert Utilities
----------------

.. currentmodule:: happyrec.utils.asserts
.. autosummary::

   assert_type
   assert_never_type
   is_typed_list
   is_typed_dict

.. autofunction:: assert_type
.. autofunction:: assert_never_type
.. autofunction:: is_typed_list
.. autofunction:: is_typed_dict

File Utilities
----------------

.. currentmodule:: happyrec.utils.file
.. autosummary::

   decompress
   checksum
   download

.. autofunction:: decompress
.. autofunction:: checksum
.. autofunction:: download

Logging Utilities
-----------------

.. currentmodule:: happyrec.utils.logger
.. autosummary::

   init_happyrec_logger
   logger

.. autofunction:: init_happyrec_logger
.. autodata:: logger

Data Preprocessing Utilities
----------------------------

.. currentmodule:: happyrec.utils.preprocessing
.. autosummary::

   load_from_jsonline
   load_from_dictline
   convert_image_to_jpeg
   convert_str_to_timestamp
   parallelize
   convert_dataframe_to_frame
   create_default_user_frame
   create_default_item_frame
   create_data
   compress

.. autofunction:: load_from_jsonline
.. autofunction:: load_from_dictline
.. autofunction:: convert_image_to_jpeg
.. autofunction:: convert_str_to_timestamp
.. autofunction:: parallelize
.. autofunction:: convert_dataframe_to_frame
.. autofunction:: create_default_user_frame
.. autofunction:: create_default_item_frame
.. autofunction:: create_data
.. autofunction:: compress
