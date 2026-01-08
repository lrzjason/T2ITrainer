declare module '*.json' {
  export interface ScriptConfig {
    name: string;
    type: 'old' | 'new';
  }

  export interface FrontendConfig {
    scripts: ScriptConfig[];
  }

  const value: FrontendConfig;
  export default value;
}