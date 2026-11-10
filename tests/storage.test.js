const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const storageScript = readFileSync(
  path.join(__dirname, "../web/storage.js"),
  "utf8",
);

function restoreDimensions(taskId, dimensionNames, dimensionValues, separate = true) {
  const changeListeners = [];
  const fields = [
    ...dimensionNames.map((name) => ({
      name,
      value: "2",
      tagName: "SELECT",
      addEventListener: (event, listener) => {
        if (event === "change") changeListeners.push(listener);
      },
    })),
    { name: "a_0_0", value: "", tagName: "INPUT" },
  ];
  const form = {
    dataset: {
      storageKey: taskId,
      resizeUrl: `/task/${taskId}`,
    },
    querySelectorAll: (selector) =>
      selector === "[data-variable-index]" ? [] : fields,
    addEventListener() {},
    querySelector() {
      return null;
    },
  };
  const saved = JSON.stringify(
    Object.fromEntries(
      dimensionNames.map((name, index) => [name, dimensionValues[index]]),
    ),
  );
  const storage = new Map([
    [
      `compmath.${taskId}.v2${separate ? ".dimensions" : ""}`,
      saved,
    ],
  ]);
  let resizeUrl;

  vm.runInNewContext(storageScript, {
    document: {
      querySelector: () => form,
    },
    localStorage: {
      getItem: (key) => storage.get(key) ?? null,
      setItem: (key, value) => storage.set(key, value),
    },
    URLSearchParams,
    FormData: class {
      constructor() {
        return fields.map((field) => [field.name, field.value]);
      }
    },
    window: {
      location: {
        replace: (url) => {
          resizeUrl = url;
        },
        assign() {},
      },
    },
  });

  return { fields, resizeUrl, storage, changeListeners };
}

test("matrix dimensions are restored independently for each task", () => {
  const simplex = restoreDimensions(
    "simplex",
    ["variables", "constraints"],
    ["3", "4"],
  );
  const dualSimplex = restoreDimensions(
    "dual_simplex",
    ["variables", "constraints"],
    ["5", "6"],
    false,
  );
  const transport = restoreDimensions(
    "transport",
    ["warehouses", "customers"],
    ["7", "8"],
  );

  assert.equal(simplex.fields[0].value, "3");
  assert.equal(simplex.fields[1].value, "4");
  assert.match(simplex.resizeUrl, /variables=3/);
  assert.match(simplex.resizeUrl, /constraints=4/);
  simplex.fields[0].value = "7";
  simplex.changeListeners[0]();
  assert.deepEqual(
    JSON.parse(simplex.storage.get("compmath.simplex.v2.dimensions")),
    { variables: "7", constraints: "4" },
  );

  assert.equal(dualSimplex.fields[0].value, "5");
  assert.equal(dualSimplex.fields[1].value, "6");
  assert.match(dualSimplex.resizeUrl, /variables=5/);
  assert.match(dualSimplex.resizeUrl, /constraints=6/);

  assert.equal(transport.fields[0].value, "7");
  assert.equal(transport.fields[1].value, "8");
  assert.match(transport.resizeUrl, /warehouses=7/);
  assert.match(transport.resizeUrl, /customers=8/);
});
