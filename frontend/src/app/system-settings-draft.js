export const systemSettingsDraftEnabled = import.meta.env.VITE_SYSTEM_SETTINGS_DRAFT_ENABLED === 'true';
export const systemSettingsApplicationEnabled =
  systemSettingsDraftEnabled && import.meta.env.VITE_SYSTEM_SETTINGS_APPLICATION_ENABLED === 'true';
