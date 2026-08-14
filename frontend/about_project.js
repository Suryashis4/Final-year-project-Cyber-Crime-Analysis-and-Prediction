function AboutSection({ eyebrow, title, description, children }) {
  return (
    <section className="section-block">
      <div className="section-heading">
        {eyebrow && <span>{eyebrow}</span>}
        <div>
          <h3>{title}</h3>
          {description && <p>{description}</p>}
        </div>
      </div>
      {children}
    </section>
  );
}

function AboutProfilePhoto({ src, alt }) {
  const [failed, setFailed] = React.useState(false);
  return (
    <div className="about-photo">
      {!failed ? (
        <img src={src} alt={alt} onError={() => setFailed(true)} />
      ) : (
        <span className="about-photo-fallback">{alt.charAt(0)}</span>
      )}
    </div>
  );
}

function AboutProject() {
  const [zoomCertificate, setZoomCertificate] = React.useState(false);

  React.useEffect(() => {
    const onKeyDown = (event) => {
      if (event.key === "Escape") setZoomCertificate(false);
    };
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      setZoomCertificate(false);
    };
  }, []);

  const members = [
    { image: TEAM_ASSETS.arnav, name: "Arnav Biswas" },
    { image: TEAM_ASSETS.ayandeep, name: "Ayandeep Roy" },
    { image: TEAM_ASSETS.suryashis, name: "Suryashis Banerjee" },
    { image: TEAM_ASSETS.rimi, name: "Rimi Dutta" },
  ];

  return (
    <div className="grid about-page">
      <AboutSection
        eyebrow="About"
        title="Project Information"
        description="Research overview for this final year project."
      >
        <div className="card about-info-card">
          <div className="about-info-grid">
            <div>
              <span className="about-label">Project Title</span>
              <h3>A Hybrid Machine Learning Framework for Geospatial Cyber Crime Prediction and Demographic Pattern Analysis</h3>
            </div>
          </div>
        </div>
      </AboutSection>

      <AboutSection eyebrow="Guide" title="Project Guide" description="Academic mentor for this research work.">
        <div className="card about-profile-card about-guide-card">
          <AboutProfilePhoto src={TEAM_ASSETS.guide} alt="Dr. Anupam Mukherjee" />
          <div>
            <h3>Dr. Anupam Mukherjee</h3>
            <p className="about-role">Project Guide</p>
            <p className="muted">Head of the Department, Department of Computer Science &amp; Engineering, Siliguri Institute of Technology</p>
          </div>
        </div>
      </AboutSection>

      <AboutSection
        eyebrow="Team"
        title="Project Members"
        description="Contributors responsible for machine learning, analysis, and project documentation."
      >
        <div className="about-team-grid">
          {members.map((member) => (
            <div className="card about-profile-card about-team-card" key={member.name}>
              <AboutProfilePhoto src={member.image} alt={member.name} />
              <h3>{member.name}</h3>
            </div>
          ))}
        </div>
      </AboutSection>

      <AboutSection
        eyebrow="Certificate"
        title="Project Certificate"
        description="Official project certificate for this research submission."
      >
        <div className="card about-certificate-card">
          <button type="button" className="about-certificate-preview" onClick={() => setZoomCertificate(true)}>
            <img src={TEAM_ASSETS.certificate} alt="Project Certificate" />
          </button>
          <p className="muted table-note">Click the certificate to zoom and view the full image.</p>
        </div>
      </AboutSection>

      {zoomCertificate && (
        <div className="about-cert-modal" onClick={() => setZoomCertificate(false)} role="presentation">
          <div className="about-cert-modal-inner" onClick={(event) => event.stopPropagation()}>
            <button type="button" className="about-cert-close" onClick={() => setZoomCertificate(false)}>
              Close
            </button>
            <img src={TEAM_ASSETS.certificate} alt="Project Certificate enlarged view" />
          </div>
        </div>
      )}
    </div>
  );
}
