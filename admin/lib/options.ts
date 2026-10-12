import type { GraphOptions, OptionProperty, OptionsSchema, SetupReport } from './types';

export interface OptionField extends OptionProperty {
  name: keyof GraphOptions;
  label: string;
}

const LABELS: Record<string, string> = {
  marble: 'World generation',
  people: 'Maximum people',
  all_people: 'Use tracked-person reconstruction',
  objects: 'Detect thrown objects',
  finetune: 'Fine-tune the generated world',
};

/** Only offer choices that this API supports, including its bounds and descriptions. */
export function optionFields(schema: OptionsSchema): OptionField[] {
  return Object.entries(schema.properties)
    .filter(([name, field]) => name in LABELS && !field.deprecated)
    .map(([name, field]) => ({
      ...field,
      name: name as keyof GraphOptions,
      label: LABELS[name] ?? field.title ?? name,
    }));
}

export function optionsFromSchema(schema: OptionsSchema): GraphOptions {
  const entries = optionFields(schema).map((field) => {
    if (field.default === undefined) {
      throw new Error(`The API did not provide a default for ${field.label}.`);
    }
    return [field.name, field.default];
  });
  const defaults = Object.fromEntries(entries);
  for (const name of Object.keys(LABELS)) {
    if (!(name in defaults))
      throw new Error(`The API does not describe the ${LABELS[name]} option.`);
  }
  return defaults as unknown as GraphOptions;
}

export function setupBlocked(report: SetupReport): boolean {
  return report.status === 'blocked' || report.checks.some((check) => check.status === 'blocked');
}
