Chunking
========

Another common use case for distributed arrays is breaking the values into chunks, and then performing operations over the subarrays. Bolt uses this kind of chunking under the hood during it's :ref:`swap` operations, but we also expose chunking directly.

