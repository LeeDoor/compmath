const form = document.querySelector("form[data-storage-key]");
if (form) {
  const key = `compmath.${form.dataset.storageKey}.v2`;
  let saved = {};
  try {
    saved = JSON.parse(localStorage.getItem(key) || "{}");
  } catch {
    saved = {};
  }
  if (!saved || typeof saved !== "object") saved = {};

  const fields = [...form.querySelectorAll("input[name], select[name]")];
  const sizeFields = new Set([
    "variables",
    "constraints",
    "warehouses",
    "customers",
  ]);
  let sizeChanged = false;
  for (const field of fields) {
    if (Object.hasOwn(saved, field.name)) {
      const previousValue = field.value;
      field.value = String(saved[field.name]);
      if (sizeFields.has(field.name)) {
        sizeChanged ||= previousValue !== field.value;
      }
    }
  }

  const save = () => {
    const values = {};
    for (const field of fields) {
      values[field.name] = field.value;
    }
    localStorage.setItem(key, JSON.stringify(values));
  };
  const resize = (values, replace = false) => {
    const query = new URLSearchParams(values);
    const url = `${form.dataset.resizeUrl}?${query}`;
    if (replace) window.location.replace(url);
    else window.location.assign(url);
  };

  if (sizeChanged) {
    const values = { ...saved };
    for (const field of fields) {
      if (!Object.hasOwn(values, field.name)) values[field.name] = field.value;
    }
    resize(values, true);
  }

  form.addEventListener("input", save);
  form.addEventListener("change", save);
  form.addEventListener("submit", save);

  for (const select of fields.filter(
    (field) => field.tagName === "SELECT" && sizeFields.has(field.name),
  )) {
    select.addEventListener("change", () => {
      save();
      resize(Object.fromEntries(new FormData(form)));
    });
  }
}
