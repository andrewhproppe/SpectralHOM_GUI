from matplotlib.ticker import ScalarFormatter


class ScalarFormatterClass(ScalarFormatter):
   def _set_format(self):
      self.format = "%1.1e"


def alphabetical_widget_list(widget_list):
    object_names = [w.objectName() for w in widget_list]
    widget_list_alphabetical = [x for _, x in sorted(zip(object_names, widget_list))]
    return widget_list_alphabetical
