

from collections import OrderedDict


class AliasDict(OrderedDict):

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.aliases = {}

    def __getitem__(self, key):
        key = self.aliases.get(key, key)
        return super().__getitem__(key)


class MultiDict(AliasDict):

    def __repr__(self):
        return 'MultiDict({})'.format(super().__repr__())

    def __str__(self):
        return 'MultiDict({})'.format(super().__str__())

    def __getitem__(self, key):
        if key == slice(None):
            key = tuple(self.keys())
        if isinstance(key, tuple):
            sup = super()
            return DictGroup(*[sup.__getitem__(item) for item in key])
        else:
            return super().__getitem__(key)


class DictGroup:

    def __new__(cls, *items):
        if all(isinstance(item, dict) for item in items):
            return object.__new__(cls)
        else:
            return items

    def __init__(self, *dicts):
        self.dicts = dicts

    def __repr__(self):
        d_reprs = [repr(d) for d in self.dicts]
        return 'DictGroup({})'.format(', '.join(d_reprs))

    def __str__(self):
        d_strs = [str(d) for d in self.dicts]
        return 'DictGroup({})'.format(', '.join(d_strs))

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return DictGroup(*[dct[item] for dct in self.dicts for item in key])
        else:
            return DictGroup(*[dct[key] for dct in self.dicts])

    def __setitem__(self, key, value):
        for dct in self.dicts:
            dct[key] = value


class Metadata(MultiDict):

    def __init__(self, domain):
        super().__init__()
        self.domain = domain
        for axis, basis in enumerate(domain.bases):
            self[basis.name] = basis.default_meta()
            #self[basis.name] = self[axis]
            self.aliases[axis] = basis.name

    # def __iter__(self):
    #     for basis in self.domain.bases:
    #         yield self[basis.name]


# class MetaCollection:

#     def __init__(self, *args):
#         self._metadata = {}
#         for arg in args:
#             self.add_metadata(arg)

#     def __getitem__(self, key):
#         return self._metadata[key]._dict_get()

#     def __setitem__(self, key, value):
#         self._metadata[key]._dict_set(value)

#     def __iter__(self):
#         yield from self._metadata

#     def __copy__(self):
#         new = MetaCollection()
#         for metadata in self._metadata.values():
#             new.add_metadata(metadata.copy())

#     def add_metadata(self, metadata):
#         self._metadata[metadata.name] = metadata


# class Metadata:

#     def __init__(self, field, axis):
#         self.field = field
#         self.axis = axis
#         self.value = self.default

#     def _dict_get(self):
#         return self.get()

#     def _dict_set(self, value):
#         self.set(value)

#     def get(self):
#         return self.value

#     def set(self, value):
#         self.value = value


# class LockedMetadata(Metadata):

#     def _dict_set(self, value):
#         raise ValueError("Cannot set LockedMetadata.")


# class Constant(Metadata):
#     name = 'constant'
#     default = False

#     def set(self, value):
#         if value == self.value:
#             return
#         if value is False:
#             self.field.require_coeff_space(self.axis)
#             self.field['c'][global_slice(1, None)] = 0
#         self.value = value


# class TransformScale(Metadata):
#     name = 'scale'
#     default = 1.

#     def set(self, value, keep_data=True):
#         if value == self.value:
#             return
#         if keep_data:
#             field.require_coeff_space(axis)
#             old_data = self.data
#         self.value = value
#         # Build new buffer
#         buffer_size = self.domain.distributor.buffer_size(new_scales)
#         self.buffer = self._create_buffer(buffer_size)
#         # Reset layout to build new data view
#         self.layout = self.layout
#         if keep_data:
#             np.copyto(self.data, old_data)


# class Parity(LockedMetadata):
#     name = 'parity'
#     default = 1

