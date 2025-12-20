import { createContext, useContext } from 'react';

export type Language = 'en' | 'cn';
export type Theme = 'dark' | 'light';

export interface FlowContextType {
    updateNodeData: (id: string, key: string, value: any) => void;
    deleteNode: (id: string) => void;
    copyNode: (id: string) => void;
    toggleLock: (id: string) => void;
    undo: () => void;
    redo: () => void;
    canUndo: boolean;
    canRedo: boolean;
    lang: Language;
    theme: Theme;
}

export const FlowContext = createContext<FlowContextType>({
    updateNodeData: () => {},
    deleteNode: () => {},
    copyNode: () => {},
    toggleLock: () => {},
    undo: () => {},
    redo: () => {},
    canUndo: false,
    canRedo: false,
    lang: 'en',
    theme: 'dark'
});

export const useFlowContext = () => useContext(FlowContext);