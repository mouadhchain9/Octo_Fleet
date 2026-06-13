import React from 'react';
import { useFleetStore } from '../../store/useFleetStore.js';

export function TopBar() {
  const {
    uiModals,
    toggleModal,
    printerStatus,
    setPrinterStatus,
    activePrinterId,
    addLogEntry
  } = useFleetStore();

  const handleStatusChange = async (newStatus) => {
    setPrinterStatus(newStatus);
    addLogEntry(`SYSTEM: Transitioning to [${newStatus.toUpperCase()}] mode...`, "SYS");

    const { AppContext } = await import('../../../app_context.js');
    const printer = AppContext.farm.printers.find(p => p.id === activePrinterId);

    // Flush the UI and 3D Canvas
    useFleetStore.getState().updateActiveJob({
       fileName: 'No file selected',
       progress: 0,
       isPrinting: false,
       isPaused: false,
       lines: "---",
       moves: "---",
       skipped: "---",
       layers: "---",
       htemp: "---",
       btemp: "---",
       filament: "---",
       kfactor: "---",
       totalTime: null
    });

    if (printer) {
       printer.filament.clear();
    }

    if (newStatus === 'standalone' || newStatus === 'disconnected' || newStatus === 'mock_replay') {
      if (printer) {
        await printer.switchMode(newStatus);
      }
    }

    toggleModal('statusOptions', false);
  };

  const getStatusClass = () => {
    if (activePrinterId === null) return '';
    if (printerStatus === 'connected') return 'status-green';
    if (printerStatus === 'standalone') return 'status-amber';
    if (printerStatus === 'mock_replay') return 'status-blue';
    return 'status-red';
  };

  return (
    <header className="top-bar">
      <div className="system-logo splash-logo-row"
        style={{
          height: '30px',
          transform: 'scale(0.8)',
          transformOrigin: 'left center',
          display: 'flex',
          alignItems: 'center',
          gap: '20px'
        }}>
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <svg className="splash-icon" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg" style={{ width: '30px', height: '30px' }}>
            <path d="M50 5 L93.3 30 V70 L50 95 L6.7 70 V30 Z" fill="#00ffc2" />
            <path d="M20 40 V75 L45 90 V55 Z" fill="#0b0f19" />
            <path d="M80 40 V75 L55 90 V55 Z" fill="#0b0f19" />
            <path d="M50 15 L25 30 L50 45 L75 30 Z" fill="#0b0f19" />
          </svg>
          <span className="splash-text" style={{ fontSize: '18px', margin: 0, padding: 0 }}>
            <span className="text-white">OCTO-</span>
            <span className="text-green">FLEET</span>
          </span>
        </div>
        <button
          onClick={() => useFleetStore.getState().toggleModal('setupGuide', true)}
          style={{
            background: '#ffffff',
            border: 'none',
            color: '#0b0f19',
            fontSize: '13px',
            padding: '6px 14px',
            cursor: 'pointer',
            borderRadius: '6px',
            fontWeight: '800',
            boxShadow: '0 0 10px rgba(255,255,255,0.3)',
            transition: 'transform 0.1s, box-shadow 0.2s'
          }}
          onMouseOver={(e) => e.currentTarget.style.boxShadow = '0 0 15px rgba(255,255,255,0.6)'}
          onMouseOut={(e) => e.currentTarget.style.boxShadow = '0 0 10px rgba(255,255,255,0.3)'}
          onMouseDown={(e) => e.currentTarget.style.transform = 'scale(0.95)'}
          onMouseUp={(e) => e.currentTarget.style.transform = 'scale(1)'}
        >
          SETUP GUIDE
        </button>
      </div>

      <div className="header-nav-center">
        <button
          className="overview-btn"
          onClick={() => {
            useFleetStore.getState().focusOverview();
            addLogEntry("Camera transitioning to Site-Wide Overview", "SYS");
          }}
          title="World Overview"
        >
          <span style={{ marginRight: '8px' }}>⬢</span> OVERVIEW
        </button>
      </div>

      <div className="header-actions">
        <div className={`status-menu-container ${activePrinterId === null ? 'disabled' : ''}`}>
          <div
            className="status-trigger"
            onClick={() => activePrinterId !== null && toggleModal('statusOptions')}
            style={{ opacity: activePrinterId === null ? 0.5 : 1, cursor: activePrinterId === null ? 'not-allowed' : 'pointer' }}
          >
            <span className={`status-text ${getStatusClass()}`}>
              {activePrinterId === null ? "NO PRINTER" : printerStatus.toUpperCase()}
            </span>
          </div>

          {uiModals.statusOptions && (
            <ul className="status-dropdown" style={{ display: 'block' }}>
              <li onClick={() => handleStatusChange('standalone')}>● STANDALONE MODE</li>
              <li onClick={() => handleStatusChange('disconnected')}>● DISCONNECT SYSTEM</li>
              <li onClick={() => handleStatusChange('mock_replay')}>● MOCK REPLAY MODE</li>
              <hr />
              <li onClick={() => {
                toggleModal('statusOptions', false);
                toggleModal('configPane', true);
              }}>⚙ CONFIGURE CONNECTION</li>
            </ul>
          )}
        </div>
        <div className="user-profile"><span>OP_01</span> <div className="avatar">M</div></div>
      </div>
    </header>
  );
}


