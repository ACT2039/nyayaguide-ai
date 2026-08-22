import React from 'react';

/**
 * Maps legal keywords in headings/list items to context-appropriate emojis.
 */
function getHeadingEmoji(text: string): string | null {
  const lower = text.toLowerCase();
  
  // Right to competitive prices / access / goods
  if (lower.includes('competitive price') || lower.includes('variety of goods') || lower.includes('access to goods') || lower.includes('price')) {
    return '💰';
  }
  // Right to be heard / representation / voice
  if (lower.includes('be heard') || lower.includes('representation') || lower.includes('interests considered') || lower.includes('voice')) {
    return '🗣️';
  }
  // Right to redressal / remedies / dispute
  if (lower.includes('redressal') || lower.includes('dispute') || lower.includes('remedy') || lower.includes('settlement')) {
    return '⚖️';
  }
  // Consumer awareness / education / info
  if (lower.includes('consumer awareness') || lower.includes('consumer education') || lower.includes('knowledge') || lower.includes('awareness')) {
    return '📚';
  }
  // Protection / safety / hazard / unfair trade
  if (lower.includes('protection against') || lower.includes('unfair trade') || lower.includes('hazardous') || lower.includes('safety') || lower.includes('dangerous goods')) {
    return '🛡️';
  }
  // Compensation / defective goods / refund
  if (lower.includes('compensation') || lower.includes('defective') || lower.includes('refund') || lower.includes('package') || lower.includes('product liability')) {
    return '📦';
  }
  // Penalties / offenses / fines / misleading ads
  if (lower.includes('penalty') || lower.includes('penalties') || lower.includes('misleading') || lower.includes('false advertisement') || lower.includes('fine')) {
    return '⚠️';
  }
  // Filing application / submission / forms / RTI
  if (lower.includes('file') || lower.includes('application') || lower.includes('submission') || lower.includes('how to apply') || lower.includes('written request')) {
    return '📝';
  }
  // Fees / payment / postal order / cash
  if (lower.includes('fee') || lower.includes('payment') || lower.includes('rupees') || lower.includes('cost')) {
    return '💳';
  }
  // Time limits / deadlines / response period / days
  if (lower.includes('timeline') || lower.includes('time limit') || lower.includes('deadline') || lower.includes('period') || lower.includes('days')) {
    return '⏱️';
  }
  // Authorities / commission / CPIO / PIO / Collector / Court
  if (lower.includes('officer') || lower.includes('commission') || lower.includes('authority') || lower.includes('collector') || lower.includes('cpio') || lower.includes('spio')) {
    return '🏛️';
  }
  // Appeals / appellate / high court
  if (lower.includes('appeal') || lower.includes('appellate')) {
    return '📜';
  }
  // Inspection / records / verification
  if (lower.includes('inspection') || lower.includes('records') || lower.includes('documents')) {
    return '🔍';
  }
  // Exemption / BPL / below poverty line
  if (lower.includes('exemption') || lower.includes('poverty line') || lower.includes('bpl')) {
    return '🏷️';
  }

  return null;
}

/**
 * Safely parses inline markdown (bold **text**, bold/italic ***text***, code `code`)
 * into React elements without showing raw markdown markers.
 */
export function renderInlineMarkdown(text: string): React.ReactNode[] {
  // Regex matches:
  // 1. **bold** or __bold__
  // 2. `code`
  // 3. plain text
  const parts: React.ReactNode[] = [];
  const regex = /(\*\*[^*]+?\*\*|__[^_]+?__|`[^`]+?`)/g;
  
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = regex.exec(text)) !== null) {
    // Push preceding plain text
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }

    const token = match[0];
    if ((token.startsWith('**') && token.endsWith('**')) || (token.startsWith('__') && token.endsWith('__'))) {
      const boldContent = token.slice(2, -2);
      parts.push(<strong key={`b-${match.index}`}>{boldContent}</strong>);
    } else if (token.startsWith('`') && token.endsWith('`')) {
      const codeContent = token.slice(1, -1);
      parts.push(<code key={`c-${match.index}`}>{codeContent}</code>);
    } else {
      parts.push(token);
    }

    lastIndex = regex.lastIndex;
  }

  // Push remaining text
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  return parts.length > 0 ? parts : [text];
}

interface FormattedLegalTextProps {
  content: string;
}

/**
 * Renders an AI legal response paragraph or list item with:
 * - Proper **bold** rendering (no visible `**`)
 * - Contextual legal emojis for numbered rights/steps
 * - Preserved structure, citations, and linebreaks
 */
export const FormattedLegalText: React.FC<FormattedLegalTextProps> = ({ content }) => {
  // Check if this block contains multiple lines (e.g. numbered items or bullet points)
  const lines = content.split('\n');

  return (
    <div className="formatted-legal-block">
      {lines.map((line, lIdx) => {
        const trimmed = line.trim();
        if (!trimmed) {
          return <div key={lIdx} className="legal-spacer" />;
        }

        // Match numbered lists like: "1. **Right to Competitive Prices**:" or "1. **Application Submission**:"
        const numberedMatch = trimmed.match(/^(\d+[\.\)])\s*(.*)$/);
        // Match bullet lists like: "- **Who can file:**" or "* **Required information:**"
        const bulletMatch = trimmed.match(/^([-*•])\s*(.*)$/);

        if (numberedMatch) {
          const numPrefix = numberedMatch[1];
          const restOfLine = numberedMatch[2];

          // Check if there is bold text inside for emoji detection
          const boldInsideMatch = restOfLine.match(/^\*\*([^*]+?)\*\*/);
          const topicCandidate = boldInsideMatch ? boldInsideMatch[1] : restOfLine.slice(0, 40);
          const emoji = getHeadingEmoji(topicCandidate);

          return (
            <div key={lIdx} className="legal-list-item numbered-item">
              <span className="list-number-badge">{numPrefix}</span>
              <div className="list-item-content">
                {emoji && <span className="legal-item-emoji" aria-hidden="true">{emoji} </span>}
                {renderInlineMarkdown(restOfLine)}
              </div>
            </div>
          );
        }

        if (bulletMatch) {
          const restOfLine = bulletMatch[2];
          const boldInsideMatch = restOfLine.match(/^\*\*([^*]+?)\*\*/);
          const topicCandidate = boldInsideMatch ? boldInsideMatch[1] : restOfLine.slice(0, 40);
          const emoji = getHeadingEmoji(topicCandidate);

          return (
            <div key={lIdx} className="legal-list-item bullet-item">
              <span className="list-bullet-badge">•</span>
              <div className="list-item-content">
                {emoji && <span className="legal-item-emoji" aria-hidden="true">{emoji} </span>}
                {renderInlineMarkdown(restOfLine)}
              </div>
            </div>
          );
        }

        // Check if paragraph itself starts with bold heading (e.g. "**Who can file:** ...")
        const topBoldMatch = trimmed.match(/^\*\*([^*]+?)\*\*/);
        const topEmoji = topBoldMatch ? getHeadingEmoji(topBoldMatch[1]) : null;

        return (
          <p key={lIdx} className="legal-paragraph">
            {topEmoji && <span className="legal-item-emoji" aria-hidden="true">{topEmoji} </span>}
            {renderInlineMarkdown(line)}
          </p>
        );
      })}
    </div>
  );
};
